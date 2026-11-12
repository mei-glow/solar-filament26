import numpy as np
import pandas as pd

from filament.p5_selector import add_candidate_context, decode_candidate_set, normalize_action_predictions


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "file_name": ["x.jpg"] * 3,
        "candidate_index": [0, 1, 2],
        "detector_score": [0.9, 0.8, 0.4],
        "semantic_support": [0.8, 0.7, 0.2],
        "raw_refined_iou": [0.8, 0.7, 0.3],
        "raw_area": [100, 100, 100],
        "refined_area": [100, 100, 100],
    })


def _pairs() -> pd.DataFrame:
    return pd.DataFrame({
        "file_name": ["x.jpg", "x.jpg"],
        "left_candidate_index": [0, 1],
        "right_candidate_index": [1, 2],
        "max_cross_variant_iou": [0.8, 0.01],
        "raw_box_iou": [0.7, 0.01],
        "predicted_duplicate_probability": [0.9, 0.1],
    })


def test_candidate_context_uses_only_local_geometry():
    output = add_candidate_context(_candidates(), _pairs())
    assert output.loc[0, "overlap_neighbor_count_050"] == 1
    assert output.loc[1, "overlap_neighbor_count_050"] == 1
    assert output.loc[2, "overlap_neighbor_count_005"] == 0
    assert np.isclose(output.loc[1, "detector_margin_to_strong_neighbor"], -0.1)


def test_graph_decoder_keeps_best_candidate_per_component():
    candidates = _candidates().assign(
        predicted_p_reject=[0.1, 0.2, 0.8],
        predicted_p_raw=[0.7, 0.6, 0.1],
        predicted_p_refined=[0.2, 0.2, 0.1],
    )
    selected = decode_candidate_set(candidates, _pairs(), 0.5, 0.5, 25, "component")
    assert [item["candidate_index"] for item in selected] == [0]
    assert selected[0]["action"] == "raw"


def test_action_probabilities_are_clipped_and_normalized():
    output = normalize_action_predictions(np.array([[-1.0, 0.2, 0.8], [0.0, 0.0, 0.0]]))
    assert np.all(output > 0)
    assert np.allclose(output.sum(axis=1), 1.0)
