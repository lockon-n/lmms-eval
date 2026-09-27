"""Reuse Qwen3.5 weights with sparse, coordinate-preserving video inputs."""

import itertools

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling


def metadata_groups(table):
    """Split the [group,Hpatch,Wpatch,block_index] rows into temporal units."""
    if table is None or table.ndim != 2 or table.shape[1] not in (4, 6) or len(table) == 0:
        raise ValueError("Sparse video requires non-empty [tokens,4 or 6] metadata")
    _, counts = torch.unique_consecutive(table[:, 0], return_counts=True)
    groups = table.split(counts.tolist())
    for group in groups:
        h, w = (int(v) for v in group[0, 1:3])
        if h <= 0 or w <= 0 or h % 2 or w % 2:
            raise ValueError("Sparse grids must be positive and merge-aligned")
        if not torch.all(group[:, 1:3] == group[0, 1:3]):
            raise ValueError("A temporal group must use one full spatial grid")
        if group.shape[1] == 6:
            reference = group[:, 4:6]
            if not torch.all(reference == reference[0]) or torch.any(reference <= 0) or torch.any(reference % 2):
                raise ValueError("Reference patch grids must be constant, positive and merge-aligned")
            if torch.any(reference[0] < group[0, 1:3]):
                raise ValueError("Reference grid must not be smaller than the final grid")
        indices = group[:, 3]
        if torch.any(indices < 0) or torch.any(indices >= h * w // 4):
            raise ValueError("Selected block index outside the full grid")
        if torch.any(indices[1:] <= indices[:-1]):
            raise ValueError("Selected blocks must be unique and in spatial order")
    return groups


def patch_positions(group):
    """Four (h,w) coordinates per selected 32x32 block, native merger order."""
    width = int(group[0, 2]) // 2
    indices = group[:, 3]
    offsets = indices.new_tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    origins = torch.stack((indices // width, indices % width), dim=-1) * 2
    return (origins[:, None, :] + offsets).reshape(-1, 2)


def selected_absolute_embeddings(visual, coordinates, h, w):
    """Native align-corners bilinear position interpolation, only at selected sites."""
    side = visual.num_grid_per_side
    y = torch.linspace(0, side - 1, h, device=coordinates.device)[coordinates[:, 0]]
    x = torch.linspace(0, side - 1, w, device=coordinates.device)[coordinates[:, 1]]
    y0, x0 = y.long(), x.long()
    y1, x1 = (y0 + 1).clamp(max=side - 1), (x0 + 1).clamp(max=side - 1)
    dy, dx = y - y0, x - x0
    indices = torch.stack((y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1))
    weights = torch.stack(((1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx))
    return (visual.pos_embed(indices) * weights[..., None]).sum(0)


class SparseVideoModelMixin:
    """Override only video features and M-RoPE; text/images and all weights stay native."""

    sparse_max_patches_per_batch = 8192

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        # Validate compact integer metadata on CPU once, rather than synchronizing
        # CUDA for each temporal group's shape/range checks.
        groups = metadata_groups(video_grid_thw.cpu() if video_grid_thw is not None else None)
        if pixel_values_videos.shape[0] != len(video_grid_thw) * 4:
            raise ValueError("Sparse video must contain four patches per output token")
        visual = self.visual
        reference = next(visual.parameters())
        chunks, offset, start = [], 0, 0
        while start < len(groups):
            end, count = start, 0
            while end < len(groups):
                next_count = len(groups[end]) * 4
                if end > start and count + next_count > self.sparse_max_patches_per_batch:
                    break
                count += next_count
                end += 1
            batch = groups[start:end]
            coords, absolute = [], []
            for group in batch:
                h, w = int(group[0, 1]), int(group[0, 2])
                group = group.to(reference.device)
                positions = patch_positions(group)
                coords.append(positions)
                absolute.append(selected_absolute_embeddings(visual, positions, h, w))
            # No unselected pixels enter patch_embed or any transformer block.
            pixels = pixel_values_videos[offset:offset + count].to(device=reference.device, dtype=reference.dtype)
            hidden = visual.patch_embed(pixels)
            hidden = hidden + torch.cat(absolute).to(hidden.dtype)
            coordinates = torch.cat(coords)
            # Qwen3.5 axial 2D RoPE: h frequencies, w frequencies, then duplicate.
            angles = (coordinates.float().unsqueeze(-1) * visual.rotary_pos_emb.inv_freq.float()).flatten(1)
            angles = torch.cat((angles, angles), dim=-1)
            position_embeddings = (angles.cos(), angles.sin())
            lengths = [len(group) * 4 for group in batch]
            cu_seqlens = torch.tensor([0] + list(itertools.accumulate(lengths)), device=hidden.device, dtype=torch.int32)
            for block in visual.blocks:
                hidden = block(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            encoded = visual.merger(hidden)
            chunks.extend(encoded.split([len(group) for group in batch]))
            offset += count
            start = end
        return BaseModelOutputWithPooling(pooler_output=chunks)

    def get_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw=None,
                       video_grid_thw=None, attention_mask=None, **kwargs):
        if video_grid_thw is None:
            return super().get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw,
                                          attention_mask=attention_mask, **kwargs)
        videos = iter(metadata_groups(video_grid_thw.cpu()))
        images = iter(image_grid_thw) if image_grid_thw is not None else iter(())
        if input_ids.shape != mm_token_type_ids.shape:
            raise ValueError("Multimodal token types must match input IDs")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("Sparse M-RoPE requires a 2D padding mask")
        output = torch.zeros((3, *input_ids.shape), device=input_ids.device, dtype=torch.long)
        deltas = []
        for b in range(len(input_ids)):
            active = torch.ones_like(input_ids[b], dtype=torch.bool) if attention_mask is None else attention_mask[b].bool()
            kinds = mm_token_type_ids[b][active].tolist()
            pieces, current = [], 0
            for kind, members in itertools.groupby(kinds):
                length = sum(1 for _ in members)
                if kind == 0:
                    positions = torch.arange(length, device=input_ids.device)[None].expand(3, -1) + current
                    current += length
                elif kind == 1:
                    grid = next(images)
                    positions = self.get_vision_position_ids(current, grid, 1, 2, device=input_ids.device)
                    current += max(int(grid[1]), int(grid[2])) // 2
                elif kind == 2:
                    group = next(videos)
                    h, w = int(group[0, 1]) // 2, int(group[0, 2]) // 2
                    indices = group[:, 3].to(input_ids.device)
                    rows, columns = indices // w, indices % w
                    extent_h, extent_w = h, w
                    if group.shape[1] == 6:
                        extent_h, extent_w = int(group[0, 4]) // 2, int(group[0, 5]) // 2
                        # Map block centers with exact integer floor arithmetic.
                        # Identity when reference and final grids are equal.
                        rows = ((2 * rows + 1) * extent_h) // (2 * h)
                        columns = ((2 * columns + 1) * extent_w) // (2 * w)
                    positions = torch.stack((torch.zeros_like(indices), rows, columns)) + current
                    # Reserve the selected coordinate system's full spatial extent.
                    current += max(extent_h, extent_w)
                else:
                    raise ValueError(f"Unknown modality {kind}")
                if positions.shape[1] != length:
                    raise ValueError("Sparse visual positions do not match placeholder count")
                pieces.append(positions)
            positions = torch.cat(pieces, dim=1)
            output[:, b, active] = positions
            deltas.append(positions.max() + 1 - int(active.sum()))
        if next(videos, None) is not None:
            raise ValueError("Unused sparse temporal groups after assigning M-RoPE")
        return output, torch.stack(deltas).unsqueeze(1)


class SparseGenerationMixin:
    def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False, input_ids=None, **kwargs):
        # Native beam expansion computes pixel lengths from dense grid products.
        if expand_size != 1 and kwargs.get("video_grid_thw") is not None:
            raise ValueError("Sparse video currently supports num_beams=1 and num_return_sequences=1")
        return super()._expand_inputs_for_generation(expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **kwargs)


def attach_sparse_video(model, max_patches_per_batch):
    """Instance-local overrides; no parameter initialization or global monkey patches."""
    vision = model.config.vision_config
    if (vision.patch_size, vision.temporal_patch_size, vision.spatial_merge_size) != (16, 2, 2):
        raise ValueError("Only Qwen3.5 patch=16, temporal=2, merge=2 is supported")
    if getattr(vision, "deepstack_visual_indexes", []):
        raise ValueError("This sparse adapter does not implement deepstack features")
    if model.config.model_type != "qwen3_5":
        raise ValueError("This entry supports dense Qwen3.5, not another architecture")
    inner = model.model
    inner.__class__ = type("SparseQwen3_5Model", (SparseVideoModelMixin, type(inner)), {})
    inner.sparse_max_patches_per_batch = max_patches_per_batch
    model.__class__ = type("SparseQwen3_5ForConditionalGeneration", (SparseGenerationMixin, type(model)), {})
