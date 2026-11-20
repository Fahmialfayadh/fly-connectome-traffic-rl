"""Unit tests for ConnectomeRecurrentCell, policy parameter parity, and lesioning."""

import numpy as np
import torch

from fly_traffic.agents.baselines import build_policy
from fly_traffic.agents.fly_policy import FlyConnectomePolicy
from fly_traffic.connectome.artifact import ConnectomeArtifact, ConnectomeMetadata


def _make_mock_artifact(num_neurons: int = 2500, seed: int = 11) -> ConnectomeArtifact:
    rng = np.random.default_rng(seed)
    exc_edges = 25000
    inh_edges = 30000

    exc_idx = torch.tensor(
        np.vstack(
            [
                rng.integers(0, num_neurons, size=exc_edges),
                rng.integers(0, num_neurons, size=exc_edges),
            ]
        ),
        dtype=torch.int64,
    )
    exc_val = torch.rand(exc_edges, dtype=torch.float32) * 0.05

    inh_idx = torch.tensor(
        np.vstack(
            [
                rng.integers(0, num_neurons, size=inh_edges),
                rng.integers(0, num_neurons, size=inh_edges),
            ]
        ),
        dtype=torch.int64,
    )
    inh_val = torch.rand(inh_edges, dtype=torch.float32) * 0.05

    cx_count = int(round(0.72 * num_neurons))
    mb_count = num_neurons - cx_count
    groups = ["CX"] * cx_count + ["MB"] * mb_count
    return ConnectomeArtifact(
        w_exc_indices=exc_idx,
        w_exc_values=exc_val,
        w_inh_indices=inh_idx,
        w_inh_values=inh_val,
        neuron_ids=list(range(num_neurons)),
        circuit_groups=groups,
        dominant_neuropils=["EB"] * cx_count + ["MB_CA_L"] * mb_count,
        dominant_nts=["ACH"] * num_neurons,
        metadata=ConnectomeMetadata(
            num_neurons=num_neurons,
            cx_count=cx_count,
            mb_count=mb_count,
            exc_edge_count=exc_edges,
            inh_edge_count=inh_edges,
            cross_region_edges=1000,
            sparsity=0.99,
            raw_spectral_radius=10.0,
            scaled_spectral_radius=0.95,
        ),
    )


def test_parameter_parity_across_architectures() -> None:
    """Verify Fly, MLP, and GRU trainable parameter counts are matched within 2%."""
    artifact = _make_mock_artifact(num_neurons=2500)
    fly = build_policy("fly", obs_dim=22, act_dim=4, artifact=artifact)
    mlp = build_policy("mlp", obs_dim=22, act_dim=4)
    gru = build_policy("gru", obs_dim=22, act_dim=4)

    fly_params = fly.count_trainable_parameters()
    mlp_params = mlp.count_trainable_parameters()
    gru_params = gru.count_trainable_parameters()

    assert fly_params == 70006
    assert abs(mlp_params - fly_params) / fly_params < 0.02
    assert abs(gru_params - fly_params) / fly_params < 0.02

    # Ensure frozen connectome matrices are not trainable nn.Parameters
    assert not fly.cell.w_exc_csr.requires_grad
    assert not fly.cell.w_inh_csr.requires_grad


def test_sparse_autograd_and_batch_sizes() -> None:
    """Verify backward gradients flow cleanly for batch sizes 1, 8, and 32."""
    artifact = _make_mock_artifact(num_neurons=400)
    policy = FlyConnectomePolicy(obs_dim=22, act_dim=4, artifact=artifact)

    for batch_size in (1, 8, 32):
        policy.zero_grad()
        obs_seq = torch.randn(4, batch_size, 22)
        dones_seq = torch.zeros(4, batch_size)
        h0 = policy.initial_hidden(batch_size)
        logits, values, h_last = policy.forward_sequence(obs_seq, h0, dones_seq)

        assert logits.shape == (4, batch_size, 4)
        assert values.shape == (4, batch_size)
        assert h_last.shape == (batch_size, 400)

        loss = logits.sum() + values.sum()
        loss.backward()

        assert policy.cell.input_projection.weight.grad is not None
        assert policy.cell.tau_logit.grad is not None
        assert policy.actor_head.weight.grad is not None
        assert policy.critic_head.weight.grad is not None


def test_pharmacology_and_lesion_masking() -> None:
    """Verify E/I gain modulation changes output and lesioning zeros targeted neurons."""
    artifact = _make_mock_artifact(num_neurons=200)
    artifact.circuit_groups = ["CX"] * 120 + ["MB"] * 80
    policy = FlyConnectomePolicy(obs_dim=22, act_dim=4, artifact=artifact)

    obs = torch.ones(1, 22)
    h0 = policy.initial_hidden(1) + 0.25

    policy.cell.set_pharmacology(1.0, 1.0)
    logits_base, _, _ = policy.forward_step(obs, h0)

    policy.cell.set_pharmacology(2.0, 0.2)
    logits_mod, _, _ = policy.forward_step(obs, h0)
    assert not torch.allclose(logits_base, logits_mod)

    removed = policy.cell.apply_region_lesion(
        circuit_groups=artifact.circuit_groups,
        target_group="CX",
        fraction=0.25,
        seed=42,
    )
    assert removed == 30
    assert policy.cell.active_neuron_count() == 170

    _, _, h_lesioned = policy.forward_step(obs, h0)
    lesioned_indices = (policy.cell.lesion_mask.squeeze(0) < 0.5)
    assert torch.all(h_lesioned[0, lesioned_indices] == 0.0)
