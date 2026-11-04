"""Tests for labelled-subset sampling and the fine-tuning helpers."""

import sys

import pytest
import torch

from src.data.cifar import NUM_CLASSES, labeled_subset
from src.models.vgg import vgg11
from src.utils.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from finetune import Classifier, augment  # noqa: E402


@pytest.fixture
def labels() -> torch.Tensor:
    return torch.arange(NUM_CLASSES).repeat_interleave(500)[torch.randperm(5000)]


def test_subsets_are_balanced_nested_and_deterministic(labels):
    small, large = labeled_subset(labels, 10, seed=0), labeled_subset(labels, 100, seed=0)
    assert torch.bincount(labels[small], minlength=NUM_CLASSES).tolist() == [1] * NUM_CLASSES
    assert torch.bincount(labels[large], minlength=NUM_CLASSES).tolist() == [10] * NUM_CLASSES
    assert set(small.tolist()) <= set(large.tolist())
    torch.testing.assert_close(small, labeled_subset(labels, 10, seed=0))
    assert not torch.equal(large, labeled_subset(labels, 100, seed=1))


def test_subset_size_must_be_divisible(labels):
    with pytest.raises(ValueError):
        labeled_subset(labels, 15, seed=0)


def test_augment_keeps_shape_and_pixels():
    images = torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8)
    out = augment(images, torch.Generator().manual_seed(0))
    assert out.shape == images.shape and out.dtype == torch.uint8
    assert set(out.unique().tolist()) <= set(images.unique().tolist()) | {0}  # crops pad with 0


def test_classifier_standardises_features():
    encoder = vgg11(layer_local=False)
    model = Classifier(encoder, torch.full((512,), 2.0), torch.full((512,), 4.0))
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, NUM_CLASSES)
    assert not model.mean.requires_grad and "mean" not in dict(model.named_parameters())
