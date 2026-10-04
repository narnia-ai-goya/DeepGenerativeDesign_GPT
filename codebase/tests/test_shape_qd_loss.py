import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE = Path(__file__).parents[1] / 'code' / 'shape_qd_loss.py'
SPEC = importlib.util.spec_from_file_location('shape_qd_loss', MODULE)
shape_qd_loss = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shape_qd_loss)


def test_contrastive_macro_morphology_targets_only_residual_cells(tmp_path):
    prototypes = np.zeros((1, 64, 64, 64), dtype=np.float32)
    # A target-only macro block creates an addition relative to the reference.
    prototypes[0, 16:20, 16:20, 16:20] = 1.0
    bank = tmp_path / 'bank.npz'
    np.savez_compressed(bank, prototypes=prototypes)

    controller = shape_qd_loss.ContrastiveMacroMorphology(
        bank, 0, 'cpu', pool=4, delta=0.10, neutral_weight=0.0)
    logits = torch.full((64, 64, 64), -8.0, requires_grad=True)
    bc = torch.zeros((64, 64, 64))
    cells = controller.prepare(logits, bc)
    assert cells['add_cells'] == 1
    assert cells['remove_cells'] == 0
    loss = controller.loss(logits, bc)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[16:20, 16:20, 16:20].mean() < 0  # gradient descent adds material


def test_local_pca_transport_reaches_requested_linear_measure_step(tmp_path):
    archive = tmp_path / 'archive.npz'
    np.savez_compressed(
        archive,
        feature_mean=np.zeros(6, dtype=np.float32),
        feature_scale=np.ones(6, dtype=np.float32),
        pca_components=np.eye(2, 6, dtype=np.float32),
        archive_scale=np.ones(2, dtype=np.float32),
        cvt_centroids=np.asarray([[1.0, -1.0]], dtype=np.float32),
    )
    controller = shape_qd_loss.LocalPcaTransport(
        archive, 0, 'cpu', dims=2, radius=0.5, ridge=1e-6,
        max_relative_latent_step=10.0)
    # Call step with a synthetic linear descriptor to test the Jacobian solve itself.
    z = torch.zeros(2, requires_grad=True)
    embedding = z
    delta, info = controller.step(z, embedding)
    assert info['clipped'] is False
    assert torch.allclose(delta, torch.tensor([0.353553, -0.353553]), atol=1e-4)
    assert abs(info['predicted_norm'] - 0.5) < 1e-4
