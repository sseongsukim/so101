"""Resolve Isaac uncolorized instance IDs to cube masks by USD prim path."""
from __future__ import annotations

import numpy as np


def resolve_cube_instance_masks(instance_ids: np.ndarray, id_to_labels: dict, env_index: int):
    """Return bit-packed held/fixed masks and auditable ID/path metadata.

    ``instance_ids`` is the uncolorized HxW integer image. Instance numbers are
    not stable; identity comes only from exact environment-scoped USD paths.
    """
    image = np.asarray(instance_ids)
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
        raise ValueError("instance IDs must be a 2D integer image")
    if not isinstance(id_to_labels, dict):
        raise ValueError("instance segmentation requires an idToLabels dictionary")
    ids_by_object, id_to_path = cube_instance_ids_by_path(id_to_labels, env_index)
    masks = np.stack([np.isin(image, sorted(ids_by_object[name])) for name in ("held_cube", "fixed_cube")])
    return np.packbits(masks.astype(np.uint8), axis=-1, bitorder="big"), id_to_path


def cube_instance_ids_by_path(id_to_labels: dict, env_index: int):
    """Resolve current annotator mapping to per-object ID sets and auditable paths."""
    if not isinstance(id_to_labels, dict):
        raise ValueError("instance segmentation requires an idToLabels dictionary")
    expected = {
        "held_cube": f"/World/envs/env_{env_index}/HeldAsset",
        "fixed_cube": f"/World/envs/env_{env_index}/FixedAsset",
    }
    ids_by_object = {name: set() for name in expected}
    id_to_path = {}
    for raw_id, raw_path in id_to_labels.items():
        instance_id, path = int(raw_id), str(raw_path)
        id_to_path[str(instance_id)] = path
        for name, root in expected.items():
            if path == root or path.startswith(root + "/"):
                ids_by_object[name].add(instance_id)
    if any(not values for values in ids_by_object.values()):
        missing = [name for name, values in ids_by_object.items() if not values]
        raise ValueError(f"instance ID map did not identify cube prim path(s): {missing}")
    return ids_by_object, id_to_path
