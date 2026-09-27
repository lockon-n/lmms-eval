"""Post-resize, pre-ViT adaptive RGB block selection for Qwen3.5."""

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor, smart_resize


def classify_ip_frames(
    transition_maes: torch.Tensor,
    *,
    threshold: float,
    max_consecutive_p_frames: int = 16,
) -> list[bool]:
    """Classify transitions while forcing an I-frame after each bounded P run."""
    if max_consecutive_p_frames < -1:
        raise ValueError("max_consecutive_p_frames must be -1 or non-negative")
    kinds = [True]
    consecutive_p = 0
    for mae in transition_maes.tolist():
        limit_reached = max_consecutive_p_frames >= 0 and consecutive_p >= max_consecutive_p_frames
        is_i = mae > threshold or limit_reached
        kinds.append(is_i)
        consecutive_p = 0 if is_i else consecutive_p + 1
    return kinds


@dataclass(frozen=True)
class SparseBlockConfig:
    rgb_mae_threshold: float = 0.17
    scan_resize_long_edge: int = 64
    max_consecutive_p_frames: int = 16
    block_metric: str = "max"
    block_threshold: float = 0.17
    thumbnail_ratio: float = 0.5
    budgets: tuple[int, ...] = (16, 32, 64)
    budget_boundaries: tuple[float, ...] = (128.0, 512.0)
    budget_alpha: float = 0.5
    max_resize_passes: int = 2
    llm_spatial_grid: str = "resized"
    p_token_ablation: str = "none"
    i_frame_selection: str = "motion"

    def __post_init__(self):
        if self.i_frame_selection not in ("motion", "uniform_equal_count"):
            raise ValueError("i_frame_selection must be motion or uniform_equal_count")
        if self.p_token_ablation not in ("none", "drop"):
            raise ValueError("p_token_ablation must be none or drop")
        if self.llm_spatial_grid not in ("resized", "reference"):
            raise ValueError("llm_spatial_grid must be resized or reference")
        if self.scan_resize_long_edge < 1 or self.max_consecutive_p_frames < -1:
            raise ValueError("Invalid scan size or consecutive-P limit")
        if self.block_metric not in ("max", "mean"):
            raise ValueError("block_metric must be max or mean")
        if not 0 <= self.thumbnail_ratio <= 1 or not 0 <= self.budget_alpha <= 1:
            raise ValueError("thumbnail_ratio and budget_alpha must lie in [0,1]")
        if self.rgb_mae_threshold < 0 or self.block_threshold < 0:
            raise ValueError("RGB thresholds must be non-negative")
        if not self.budgets or len(self.budgets) != len(self.budget_boundaries) + 1:
            raise ValueError("Need one fewer boundary than budgets")
        for values in (self.budgets, self.budget_boundaries):
            if any(v <= 0 for v in values) or any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError("Budgets and boundaries must be positive and strictly increasing")
        if self.max_resize_passes not in (1, 2):
            raise ValueError("max_resize_passes must be 1 or 2")


def aligned_size(height, width, scale):
    """Nearest 32-aligned dimensions, preserving aspect approximately, never upscaling."""
    return tuple(max(32, min(d, round(d * min(scale, 1.0) / 32) * 32)) for d in (height, width))


def block_scores(pair, metric):
    """Difference reduction over RGB channels and each complete 32x32 block."""
    _, _, h, w = pair.shape
    diff = (pair[1].float() - pair[0].float()).abs().reshape(3, h // 32, 32, w // 32, 32)
    return (diff.amax(dim=(0, 2, 4)) if metric == "max" else diff.mean(dim=(0, 2, 4))).flatten()


def select_blocks(pair, config):
    """Select RGB blocks; both resizes always read the same original RGB pair."""
    h, w = pair.shape[-2:]
    scores = block_scores(pair, config.block_metric)
    selected = (scores > config.block_threshold).nonzero().flatten()
    n, k = scores.numel(), selected.numel()
    ratio = k / n
    budget = config.budgets[0]
    mode, passes = "direct", 0
    final = pair
    demand = n * ratio**config.budget_alpha if k else 0.0
    if k > config.budgets[0]:
        budget = config.budgets[sum(demand > boundary for boundary in config.budget_boundaries)]
        mode = "thumbnail" if ratio > config.thumbnail_ratio else "local"
        scale = min(1.0, math.sqrt(budget / (n if mode == "thumbnail" else k)))
        target = aligned_size(h, w, scale)
        final = F.interpolate(pair, size=target, mode="nearest")
        passes = 1
        if mode == "thumbnail":
            selected = torch.arange(target[0] * target[1] // 1024, device=pair.device)
        else:
            selected = (block_scores(final, config.block_metric) > config.block_threshold).nonzero().flatten()
            if selected.numel() > budget and config.max_resize_passes == 2:
                target = aligned_size(h, w, scale * math.sqrt(budget / selected.numel()))
                final = F.interpolate(pair, size=target, mode="nearest")
                selected = (block_scores(final, config.block_metric) > config.block_threshold).nonzero().flatten()
                passes = 2
    return final, selected, dict(
        mode=mode, original_size=[h, w], final_size=list(final.shape[-2:]),
        total_blocks=n, initial_concerned_blocks=k, initial_ratio=ratio,
        demand=demand, target_budget=budget, resize_passes=passes,
        selected_blocks=selected.numel(), visual_tokens=selected.numel(),
        final_total_blocks=final.shape[-2] * final.shape[-1] // 1024,
        final_selected_ratio=selected.numel() / (final.shape[-2] * final.shape[-1] // 1024),
        selected_block_indices=selected.tolist(),
    )


def sparse_patchify(pair, selected):
    """Gather raw 32x32 blocks BEFORE projection; emit four native patches per block."""
    _, c, h, w = pair.shape
    # Index the block view directly: do not materialize a dense patchified image.
    blocks = pair.reshape(2, c, h // 32, 32, w // 32, 32).permute(2, 4, 0, 1, 3, 5)
    blocks = blocks[selected // (w // 32), selected % (w // 32)]
    return blocks.reshape(-1, 2, c, 2, 16, 2, 16).permute(0, 3, 5, 2, 1, 4, 6).reshape(-1, c * 2 * 16**2)


class SparseBlockVideoProcessor(Qwen3VLVideoProcessor):
    """Input frames are sampled upstream; selection happens after native resize.

    video_grid_thw is a custom [output_tokens,4] table: temporal group ID,
    full patch-grid H/W, and row-major 32x32 block index. No mutable model state.
    """

    sparse_config = SparseBlockConfig()

    def to_dict(self):
        """Keep HF processor repr/save JSON-compatible; exclude per-request caches."""
        result = super().to_dict()
        result["sparse_config"] = asdict(self.sparse_config)
        for key in ("_last_pair_plans", "_last_sparse_stats"):
            result.pop(key, None)
        return result

    def _preprocess(self, videos, do_convert_rgb=True, do_resize=True, size=None,
                    resample=PILImageResampling.BICUBIC, do_rescale=True, rescale_factor=1 / 255,
                    do_normalize=True, image_mean=None, image_std=None, patch_size=None,
                    temporal_patch_size=None, merge_size=None, return_tensors=None, **kwargs):
        if (int(patch_size or self.patch_size), int(temporal_patch_size or self.temporal_patch_size),
                int(merge_size or self.merge_size)) != (16, 2, 2):
            raise ValueError("Sparse blocks require Qwen patch=16, temporal=2, merge=2")
        size = size or self.size
        if not isinstance(size, SizeDict):
            size = SizeDict(**size)
        cfg = self.sparse_config
        payloads, tables, plans, stats = [], [], [], []
        group_id = 0
        for video in videos:
            if do_convert_rgb:
                video = self.convert_to_rgb(video[None])[0]
            if video.ndim != 4 or len(video) == 0 or video.shape[1] != 3:
                raise ValueError("Expected non-empty RGB video [T,3,H,W]")
            t, _, h, w = video.shape
            if do_resize:
                rh, rw = smart_resize(num_frames=t, height=h, width=w, temporal_factor=2, factor=32,
                                      min_pixels=size.shortest_edge, max_pixels=size.longest_edge)
                video = self.resize(video, size=SizeDict(height=rh, width=rw), resample=resample)
            h, w = video.shape[-2:]
            if h % 32 or w % 32:
                raise ValueError("Upstream frames must be 32-aligned when do_resize=False")
            # Raw input contract is the native Qwen RGB range (normally 0..255).
            raw_scale = float(rescale_factor) if do_rescale else 1.0
            scan_scale = min(1.0, cfg.scan_resize_long_edge / max(h, w))
            scan_size = (max(1, round(h * scan_scale)), max(1, round(w * scan_scale)))
            scan = torch.cat([F.interpolate(f[None].float(), size=scan_size, mode="nearest") for f in video]) * raw_scale
            maes = (scan[1:] - scan[:-1]).abs().mean(dim=(1, 2, 3))
            kinds = classify_ip_frames(maes, threshold=cfg.rgb_mae_threshold,
                                       max_consecutive_p_frames=cfg.max_consecutive_p_frames)
            motion_i_indices = [i for i, is_i in enumerate(kinds) if is_i]
            if cfg.i_frame_selection == "uniform_equal_count":
                # Left edges of K equal temporal bins: first frame remains I.
                # Gaps (including the tail) are <= ceil(T/K), so the original
                # forced-I count also preserves the configured consecutive-P cap.
                count = len(motion_i_indices)
                uniform_indices = [(i * t) // count for i in range(count)]
                kinds = [False] * t
                for i in uniform_indices:
                    kinds[i] = True
                if sum(kinds) != count:
                    raise ValueError("Uniform I selection changed the reference I count")
            p_run = max_p_run = 0
            for is_i in kinds:
                p_run = 0 if is_i else p_run + 1
                max_p_run = max(max_p_run, p_run)
            if cfg.max_consecutive_p_frames >= 0 and max_p_run > cfg.max_consecutive_p_frames:
                raise ValueError("I selection exceeds the consecutive-P limit")
            plan, frame_stats = [], []
            index = 0
            while index < t:
                current = index
                if kinds[index]:
                    second = min(index + 1, t - 1)
                    first = index
                    pair = torch.stack((video[first], video[second])).float() * raw_scale
                    selected = torch.arange(h * w // 1024, device=video.device)
                    record = dict(mode="I", original_size=[h, w], final_size=[h, w],
                                  visual_tokens=selected.numel(), selected_blocks=selected.numel())
                    index += 2 if second != first and kinds[second] else 1
                else:
                    first, second = index - 1, index
                    if cfg.p_token_ablation == "drop":
                        frame_stats.append(dict(frame_index=index, kind="P", mode="drop",
                            pair_indices=[first, second], original_size=[h, w],
                            final_size=None, selected_blocks=0, visual_tokens=0))
                        index += 1
                        continue
                    pair = torch.stack((video[first], video[second])).float() * raw_scale
                    pair, selected, record = select_blocks(pair, cfg)
                    index += 1
                record.update(frame_index=current, kind="I" if kinds[current] else "P",
                              pair_indices=[first, second])
                frame_stats.append(record)
                if selected.numel() == 0:
                    continue
                # Already rescaled for comparisons; use the native normalize operation once.
                pair = self.rescale_and_normalize(pair[None], False, 1.0, do_normalize,
                                                 image_mean, image_std)[0]
                payloads.append(sparse_patchify(pair, selected))
                reference_grid = cfg.llm_spatial_grid == "reference"
                table = torch.empty((selected.numel(), 6 if reference_grid else 4), dtype=torch.long, device=video.device)
                table[:, 0] = group_id
                table[:, 1] = pair.shape[-2] // 16
                table[:, 2] = pair.shape[-1] // 16
                table[:, 3] = selected
                if reference_grid:
                    # Full post-Qwen-resize grid, before any adaptive P resize.
                    table[:, 4] = h // 16
                    table[:, 5] = w // 16
                tables.append(table)
                plan.append((first, second, selected.numel()))
                group_id += 1
            plans.append(plan)
            stats.append(dict(config=asdict(cfg), frames=t, work_size=[h, w],
                              motion_I=len(motion_i_indices), motion_i_indices=motion_i_indices,
                              selected_i_indices=[i for i, is_i in enumerate(kinds) if is_i],
                              actual_max_consecutive_p_frames=max_p_run,
                              I=sum(kinds), P=t - sum(kinds), frame_kinds=["I" if k else "P" for k in kinds],
                              transition_maes=maes.tolist(), groups=frame_stats,
                              visual_tokens=sum(p[2] for p in plan)))
        self._last_pair_plans, self._last_sparse_stats = plans, stats
        return BatchFeature(data=dict(pixel_values_videos=torch.cat(payloads),
                                      video_grid_thw=torch.cat(tables)), tensor_type=return_tensors)


class SparseBlockProcessor(Qwen3VLProcessor):
    def replace_video_token(self, video_inputs, video_idx):
        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.fps is None or metadata.fps <= 0:
            raise ValueError("Sparse video requires source FPS for real timestamps")
        source = metadata.frames_indices
        parts = []
        for first, second, count in self.video_processor._last_pair_plans[video_idx]:
            timestamp = (float(source[first]) + float(source[second])) / (2 * metadata.fps)
            parts.append(f"<{timestamp:.1f} seconds>" + self.vision_start_token + self.video_token * count + self.vision_end_token)
        return "".join(parts)

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        if video_sizes is not None:
            raise ValueError("Sparse video token counts require RGB content, not just video dimensions")
        return super()._get_num_multimodal_tokens(image_sizes=image_sizes, **kwargs)
