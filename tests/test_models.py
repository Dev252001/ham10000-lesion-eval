"""
tests/test_models.py
--------------------
Tests for skinlesion.transforms and skinlesion.models.

All model tests use pretrained=False (weights=None) to avoid network access.
"""
from __future__ import annotations

import pytest

from skinlesion.data import CLASSES

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

import pandas as pd

from skinlesion.models import (
    BaselineCNN,
    build_model,
    compute_class_weights,
    count_params,
)
from skinlesion.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    eval_transform,
    train_transform,
)


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_imagenet_mean_length(self):
        assert len(IMAGENET_MEAN) == 3

    def test_imagenet_std_length(self):
        assert len(IMAGENET_STD) == 3

    def test_imagenet_mean_values(self):
        # Well-known ImageNet statistics – fixed, never dataset-derived
        assert IMAGENET_MEAN == pytest.approx((0.485, 0.456, 0.406), abs=1e-6)

    def test_imagenet_std_values(self):
        assert IMAGENET_STD == pytest.approx((0.229, 0.224, 0.225), abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: transforms
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_train_transform_output_shape(self):
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        t = train_transform(image_size=64)
        out = t(img)
        assert out.shape == torch.Size([3, 64, 64])

    def test_eval_transform_output_shape(self):
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        t = eval_transform(image_size=64)
        out = t(img)
        assert out.shape == torch.Size([3, 64, 64])

    def test_eval_transform_dtype(self):
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        t = eval_transform(image_size=32)
        out = t(img)
        assert out.dtype == torch.float32

    def test_train_transform_different_sizes(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100))
        for sz in [32, 64, 128, 224]:
            t = train_transform(image_size=sz)
            out = t(img)
            assert out.shape == torch.Size([3, sz, sz])


# ---------------------------------------------------------------------------
# Tests: output shapes for all three models
# ---------------------------------------------------------------------------

class TestModelOutputShapes:
    """All models tested at 64x64 with pretrained=False (no network access)."""

    @pytest.fixture(params=["baseline_cnn", "resnet18", "efficientnet_b0"])
    def model(self, request):
        m = build_model(request.param, pretrained=False)
        m.eval()
        return m

    def test_output_shape_7(self, model):
        x = torch.zeros(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == torch.Size([2, 7])

    def test_output_dtype_float32(self, model):
        x = torch.zeros(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: BaselineCNN architecture
# ---------------------------------------------------------------------------

class TestBaselineCNN:
    @pytest.fixture(scope="class")
    def model(self):
        return BaselineCNN()

    def test_has_features(self, model):
        assert hasattr(model, "features")

    def test_has_classifier(self, model):
        assert hasattr(model, "classifier")

    def test_output_shape(self, model):
        model.eval()
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == torch.Size([1, 7])


# ---------------------------------------------------------------------------
# Tests: count_params matches manual sum
# ---------------------------------------------------------------------------

class TestCountParams:
    def test_count_params_baseline(self):
        model = BaselineCNN()
        manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count_params(model) == manual

    def test_count_params_resnet18(self):
        model = build_model("resnet18", pretrained=False)
        manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count_params(model) == manual

    def test_count_params_efficientnet(self):
        model = build_model("efficientnet_b0", pretrained=False)
        manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count_params(model) == manual

    def test_count_params_positive(self):
        model = build_model("resnet18", pretrained=False)
        assert count_params(model) > 0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model name"):
            build_model("unknown_arch", pretrained=False)


# ---------------------------------------------------------------------------
# Tests: compute_class_weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:
    """
    Known table:
      N = 10, C = 7
      mel: 4, bcc: 3, akiec: 1, bkl: 1, df: 1, nv: 0 (absent), vasc: 0 (absent)
    Expected weights (N / (C * n_c)):
      akiec = 10 / (7 * 1) = 10/7
      bcc   = 10 / (7 * 3) = 10/21
      bkl   = 10 / (7 * 1) = 10/7
      df    = 10 / (7 * 1) = 10/7
      mel   = 10 / (7 * 4) = 10/28 = 5/14
      nv    = 0.0  (absent)
      vasc  = 0.0  (absent)
    CLASSES order: akiec bcc bkl df mel nv vasc  (sorted)
    """

    @pytest.fixture(scope="class")
    def train_df(self):
        rows = (
            [{"dx": "mel"}] * 4
            + [{"dx": "bcc"}] * 3
            + [{"dx": "akiec"}] * 1
            + [{"dx": "bkl"}] * 1
            + [{"dx": "df"}] * 1
        )
        return pd.DataFrame(rows)

    @pytest.fixture(scope="class")
    def weights(self, train_df):
        return compute_class_weights(train_df)

    def test_shape(self, weights):
        assert weights.shape == torch.Size([7])

    def test_dtype(self, weights):
        assert weights.dtype == torch.float32

    def test_akiec(self, weights):
        idx = list(CLASSES).index("akiec")
        assert weights[idx].item() == pytest.approx(10 / 7, rel=1e-5)

    def test_bcc(self, weights):
        idx = list(CLASSES).index("bcc")
        assert weights[idx].item() == pytest.approx(10 / 21, rel=1e-5)

    def test_mel(self, weights):
        idx = list(CLASSES).index("mel")
        assert weights[idx].item() == pytest.approx(10 / 28, rel=1e-5)

    def test_absent_class_zero(self, weights):
        nv_idx = list(CLASSES).index("nv")
        vasc_idx = list(CLASSES).index("vasc")
        assert weights[nv_idx].item() == pytest.approx(0.0)
        assert weights[vasc_idx].item() == pytest.approx(0.0)

    def test_majority_class_gets_smaller_weight(self, weights):
        """The majority class (mel, 4 images) should have a smaller weight than
        minority classes (akiec/bkl/df, 1 image each)."""
        mel_idx = list(CLASSES).index("mel")
        akiec_idx = list(CLASSES).index("akiec")
        assert weights[mel_idx].item() < weights[akiec_idx].item()