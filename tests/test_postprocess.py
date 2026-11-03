import numpy as np

from filament.postprocess import mask_to_rle_object, pq_stats_rle, resolve_instances


def test_pq_identity_and_empty_cases():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:8, 3:9] = True
    rle = mask_to_rle_object(mask)
    assert pq_stats_rle([rle], [rle])["pq"] == 1.0
    assert pq_stats_rle([], [rle])["fp"] == 1
    assert pq_stats_rle([rle], [])["fn"] == 1
    assert pq_stats_rle([], [])["pq"] == 0.0


def test_resolved_instances_never_overlap():
    masks = np.zeros((2, 16, 16), dtype=bool)
    masks[0, 2:10, 2:10] = True
    masks[1, 6:14, 6:14] = True
    instances = resolve_instances(masks, np.asarray([0.9, 0.8]), conf=0.1, min_area=1, duplicate_iou=0.95)
    assert len(instances) == 2
    assert not np.logical_and(instances[0], instances[1]).any()
