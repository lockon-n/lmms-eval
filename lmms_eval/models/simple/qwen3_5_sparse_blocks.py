"""Qwen3.5 eval with adaptive, pre-ViT RGB block selection (no PairDiff network)."""

from datetime import datetime, timezone
import json
from pathlib import Path

from loguru import logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.sparse_blocks.modeling import attach_sparse_video
from lmms_eval.models.model_utils.sparse_blocks.processing import SparseBlockConfig, SparseBlockProcessor, SparseBlockVideoProcessor
from lmms_eval.models.simple.qwen3_5 import Qwen3_5


def _values(value, cast):
    if isinstance(value, (int, float)):
        return (cast(value),)
    if value == "":
        return ()
    return tuple(cast(v) for v in (value.split(":") if isinstance(value, str) else value))


@register_model("qwen3_5_sparse_blocks")
class Qwen3_5_SparseBlocks(Qwen3_5):
    def __init__(self, pretrained="Qwen/Qwen3.5-9B", processor=None, do_resize=True,
                 rgb_mae_threshold=0.17, scan_resize_long_edge=64, max_consecutive_p_frames=16,
                 block_metric="max", block_threshold=0.17, thumbnail_ratio=0.5,
                 budgets="16:32:64", budget_boundaries=None, budget_alpha=0.5,
                 max_resize_passes=2, vision_max_patches_per_batch=8192, sparse_stats_dir=None,
                 llm_spatial_grid="resized", p_token_ablation="none", i_frame_selection="motion", **kwargs):
        parsed_budgets = _values(budgets, int)
        if budget_boundaries is None:
            budget_boundaries = () if len(parsed_budgets) == 1 else (128, 512)
        self.sparse_config = SparseBlockConfig(
            rgb_mae_threshold=float(rgb_mae_threshold), scan_resize_long_edge=int(scan_resize_long_edge),
            max_consecutive_p_frames=int(max_consecutive_p_frames), block_metric=block_metric,
            block_threshold=float(block_threshold), thumbnail_ratio=float(thumbnail_ratio),
            budgets=parsed_budgets, budget_boundaries=_values(budget_boundaries, float),
            budget_alpha=float(budget_alpha), max_resize_passes=int(max_resize_passes),
            llm_spatial_grid=llm_spatial_grid, p_token_ablation=p_token_ablation,
            i_frame_selection=i_frame_selection)
        self.sparse_max_patches_per_batch = int(vision_max_patches_per_batch)
        if self.sparse_max_patches_per_batch < 4:
            raise ValueError("vision_max_patches_per_batch must be >=4")
        self.do_resize = do_resize
        super().__init__(pretrained=pretrained, **kwargs)
        if self.batch_size_per_gpu != 1:
            raise ValueError("Sparse blocks currently require batch_size=1")
        source = processor or pretrained
        vp = SparseBlockVideoProcessor.from_pretrained(source, max_frames=self.max_num_frames)
        vp.sparse_config = self.sparse_config
        self.processor = SparseBlockProcessor.from_pretrained(source, video_processor=vp)
        root = sparse_stats_dir
        if root is None:
            root = Path("logs") / ("sparse_blocks_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        self.sparse_stats_path = Path(root) / f"sparse_blocks_rank{self.rank}.jsonl"
        self.sparse_stats_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Sparse pre-ViT blocks: config={}, do_resize={}, statistics={}",
                    self.sparse_config, self.do_resize, self.sparse_stats_path)

    def _configure_loaded_model(self):
        attach_sparse_video(self._model, self.sparse_max_patches_per_batch)

    def _processor_video_call_kwargs(self):
        return {"do_resize": self.do_resize}

    def _preprocess_chunk(self, chunk):
        # The base evaluator has one preprocessing worker; do not transfer mutable
        # processor state to forward. Metadata tensors travel with each request.
        self.processor.video_processor._last_sparse_stats = []
        result = super()._preprocess_chunk(chunk)
        inputs = result[0]
        stats = self.processor.video_processor._last_sparse_stats
        if stats:
            visual_tokens = int((inputs["input_ids"] == self.processor.video_token_id).sum())
            if visual_tokens != sum(v["visual_tokens"] for v in stats):
                raise ValueError("Sparse statistics disagree with actual video placeholders")
            records = [dict(doc_id=item[3], task=item[4], split=item[5]) for item in chunk]
            with self.sparse_stats_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(requests=records, pretrained=self.config._name_or_path,
                    do_resize=self.do_resize, visual_tokens=visual_tokens, videos=stats), ensure_ascii=False) + "\n")
        return result
