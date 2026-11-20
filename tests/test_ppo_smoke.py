"""Integration smoke tests for PPOTrainer and checkpoint persistence."""

from pathlib import Path

import torch

from fly_traffic.agents.baselines import load_policy_checkpoint
from fly_traffic.config import ConnectomeConfig, PPOConfig, TrafficEnvConfig
from fly_traffic.training.ppo_trainer import PPOTrainer
from tests.test_policies import _make_mock_artifact


def test_ppo_smoke_train_and_reload(tmp_path: Path) -> None:
    """Run short 2-update PPO training for Fly and GRU and verify finite updates & reload."""
    conn_path = tmp_path / "mock_connectome.pt"
    ckpt_dir = tmp_path / "checkpoints"

    artifact = _make_mock_artifact(num_neurons=100)
    artifact.save(conn_path)

    ppo_cfg = PPOConfig(
        total_timesteps=256,  # 2 updates * (4 envs * 32 steps)
        num_envs=4,
        rollout_steps=32,
        sequence_chunk_len=8,
        num_minibatches=4,
        update_epochs=2,
        checkpoints_dir=ckpt_dir,
        num_cpu_threads=2,
    )
    env_cfg = TrafficEnvConfig(max_steps=40, profile_name="normal")
    conn_cfg = ConnectomeConfig(output_path=conn_path)

    trainer = PPOTrainer(
        ppo_config=ppo_cfg,
        env_config=env_cfg,
        connectome_config=conn_cfg,
    )

    for model_type in ("fly", "gru"):
        policy, history = trainer.train(
            model_type=model_type,
            artifact=artifact if model_type == "fly" else None,
            verbose=False,
        )
        assert len(history.updates) == 2
        assert all(torch.isfinite(torch.tensor(v)) for v in history.policy_loss)

        ckpt_file = ckpt_dir / f"{model_type}_ppo.pt"
        assert ckpt_file.exists()

        reloaded, payload = load_policy_checkpoint(
            ckpt_file,
            artifact=artifact if model_type == "fly" else None,
        )
        assert payload["model_type"] == model_type
        assert reloaded.count_trainable_parameters() == policy.count_trainable_parameters()
