import asyncio
import pathlib

from etils import epath
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.training import checkpoints


def test_callback_handler_uses_contracted_signal_future(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setattr(checkpoints.jax, "process_index", lambda: 0)

    def save_assets(directory):
        asset_dir = directory / "train"
        asset_dir.mkdir(parents=True)
        (asset_dir / "norm_stats.json").write_text("{}")

    commit_futures = asyncio.run(
        checkpoints.CallbackHandler().async_save(epath.Path(tmp_path), checkpoints.CallbackSave(save_assets))
    )
    assert len(commit_futures) == 1
    assert isinstance(commit_futures[0], future.CommitFutureAwaitingContractedSignals)

    commit_futures[0].result()

    assert (tmp_path / "train" / "norm_stats.json").is_file()


def test_composite_checkpoint_save_and_restore_on_cpu(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setattr(checkpoints.jax, "process_index", lambda: 0)
    manager = ocp.CheckpointManager(
        tmp_path,
        item_handlers={
            "assets": checkpoints.CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=60),
        ),
    )

    def save_assets(directory):
        asset_dir = directory / "train"
        asset_dir.mkdir(parents=True)
        (asset_dir / "norm_stats.json").write_text('{"synthetic": true}')

    train_state = {"step": np.asarray(1000, dtype=np.int32)}
    params = {"params": {"weight": np.arange(4, dtype=np.float32).reshape(2, 2)}}
    manager.save(
        1000,
        items={
            "assets": save_assets,
            "train_state": train_state,
            "params": params,
        },
    )
    manager.wait_until_finished()

    checkpoint_dir = tmp_path / "1000"
    assert (checkpoint_dir / "assets" / "train" / "norm_stats.json").is_file()
    assert not list(tmp_path.glob("*.orbax-checkpoint-tmp-*"))

    restored = manager.restore(
        1000,
        items={
            "train_state": {"step": np.asarray(0, dtype=np.int32)},
            "params": {"params": {"weight": np.zeros((2, 2), dtype=np.float32)}},
        },
    )
    np.testing.assert_array_equal(restored["train_state"]["step"], train_state["step"])
    np.testing.assert_array_equal(restored["params"]["params"]["weight"], params["params"]["weight"])
    manager.close()
