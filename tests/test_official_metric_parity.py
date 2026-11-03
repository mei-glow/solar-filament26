import numpy as np

from filament.postprocess import pq_one


def test_official_metric_uses_strict_greater_than_half():
    gt = np.zeros((2, 3), bool); gt[0, :] = True
    prediction = np.zeros((2, 3), bool); prediction[:, :2] = True
    # intersection=2, union=4, exactly 0.5: official notebook does not match.
    pq, tp, fp, fn = pq_one([gt], [prediction])
    assert (pq, tp, fp, fn) == (0.0, 0, 1, 1)


def test_official_pooled_pq_is_sum_iou_over_pooled_denominator():
    gt_a = np.array([[1, 1], [0, 0]], bool)
    pred_a = gt_a.copy()
    gt_b = np.array([[1, 1], [1, 1]], bool)
    # No prediction must be represented by an empty list.  A list containing
    # an all-zero mask is still one predicted instance and correctly adds FP=1.
    first = pq_one([gt_a], [pred_a]); second = pq_one([gt_b], [])
    iou_sum = 1.0
    denominator = first[1] + second[1] + .5 * (first[2] + second[2]) + .5 * (first[3] + second[3])
    assert iou_sum / denominator == 2 / 3
