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


def test_equivalent_labels_interpolates_in_log_space():
    from src.analysis.data_equivalence import data_equivalence_table, equivalent_labels

    counts, accs = [10, 100, 1000], [0.2, 0.4, 0.6]
    assert equivalent_labels(0.4, counts, accs) == pytest.approx(100)
    assert equivalent_labels(0.3, counts, accs) == pytest.approx(10**1.5)  # halfway in log(n)
    assert equivalent_labels(0.1, counts, accs) != equivalent_labels(0.1, counts, accs)  # nan
    assert equivalent_labels(0.9, counts, accs) == float("inf")
    # Non-monotone baselines use the running maximum.
    assert equivalent_labels(0.5, [10, 100, 1000], [0.3, 0.6, 0.55]) == pytest.approx(
        10**1.6667, rel=1e-3
    )
    row = data_equivalence_table([10], [0.4], counts, accs)[0]
    assert row["equivalent_n"] == pytest.approx(100) and row["ratio"] == pytest.approx(10)
