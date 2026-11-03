import numpy as np

from filament.fusion import cluster_detector_candidates, fuse_instances


def test_detector_clustering_merges_duplicate_models():
    first = np.zeros((32, 32), dtype=bool)
    second = np.zeros_like(first)
    first[5:15, 5:20] = True
    second[6:16, 5:20] = True
    clusters = cluster_detector_candidates(
        [(first[None], np.asarray([0.8])), (second[None], np.asarray([0.9]))],
        confidence=0.2,
        cluster_iou=0.5,
    )
    assert len(clusters) == 1
    assert clusters[0].support == 2
    assert clusters[0].score == 0.9


def test_seeded_watershed_produces_non_overlapping_instances():
    semantic = np.zeros((3, 64, 64), dtype=np.float32)
    semantic[0, 20:44, 5:59] = 0.9
    masks = np.zeros((2, 64, 64), dtype=bool)
    masks[0, 24:40, 8:28] = True
    masks[1, 24:40, 36:56] = True
    result = fuse_instances(
        semantic, [(masks, np.asarray([0.9, 0.85]))],
        detector_confidence=0.2, semantic_threshold=0.5, min_area=10,
        expansion=1.0, cluster_iou=0.8,
    )
    assert len(result) == 2
    assert not np.logical_and(result[0][0], result[1][0]).any()
