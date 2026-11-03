import numpy as np

from filament.semantic_data import build_features, radial_map


def test_semantic_features_have_three_bounded_channels():
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    features = build_features(image, "radial", radial_map(64, 64))
    assert features.shape == (3, 64, 64)
    assert np.isfinite(features).all()
    assert 0 <= features.min() <= features.max() <= 1
