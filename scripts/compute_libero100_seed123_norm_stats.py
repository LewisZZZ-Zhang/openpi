"""Compute native pi0-FAST stats using the exact RICL LIBERO-100 train distribution."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from openpi.shared import normalize
from openpi.training.libero_manifest_dataset import iter_training_trajectories


def _ricl_native_stats(ricl_stats_path: pathlib.Path) -> dict[str, normalize.NormStats]:
    payload = json.loads(ricl_stats_path.read_text(encoding="utf-8"))["norm_stats"]
    return {
        "state": normalize.NormStats(**payload["query_state"]),
        "actions": normalize.NormStats(**payload["query_actions"]),
    }


def _assert_matches_ricl(native_stats: dict, expected: dict[str, normalize.NormStats]) -> None:
    for key in ("state", "actions"):
        for statistic in ("mean", "std", "q01", "q99"):
            actual_values = np.asarray(getattr(native_stats[key], statistic))
            expected_values = np.asarray(getattr(expected[key], statistic))
            # RunningStats estimates quantiles with a dynamically resized
            # histogram, so recomputation can differ by roughly one bin.
            tolerance = 1e-3 if statistic in ("q01", "q99") else 1e-6
            np.testing.assert_allclose(actual_values, expected_values, rtol=0.0, atol=tolerance)


def compute_norm_stats(
    corpus_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    ricl_stats_path: pathlib.Path | None,
) -> None:
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    trajectory_count = 0
    frame_count = 0
    for states, actions in iter_training_trajectories(corpus_dir, include_context=True):
        state_stats.update(states)
        action_stats.update(actions)
        trajectory_count += 1
        frame_count += len(actions)

    stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    if ricl_stats_path is not None:
        canonical_stats = _ricl_native_stats(ricl_stats_path)
        _assert_matches_ricl(stats, canonical_stats)
        # Save the exact values consumed by RICL, with native pi0-FAST keys.
        stats = canonical_stats
        print(f"Verified native stats against {ricl_stats_path.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(output_dir, stats)
    print(
        f"Wrote stats from {trajectory_count} trajectories and {frame_count} frames "
        f"to {(output_dir / 'norm_stats.json').resolve()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        type=pathlib.Path,
        default=pathlib.Path("../../data/processed/ricl_libero100_70_30"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("assets/pi0_fast_libero100_seed123/libero100_seed123_70_30"),
    )
    parser.add_argument(
        "--ricl-stats-path",
        type=pathlib.Path,
        default=pathlib.Path("../ricl_openpi/assets/libero_ricl/norm_stats.json"),
    )
    parser.add_argument("--no-verify-ricl", action="store_true")
    args = parser.parse_args()
    compute_norm_stats(
        args.corpus_dir.expanduser(),
        args.output_dir.expanduser(),
        ricl_stats_path=None if args.no_verify_ricl else args.ricl_stats_path.expanduser(),
    )


if __name__ == "__main__":
    main()
