"""Serve a trained Pi0.5 VICTR policy for LIBERO rollout."""

from __future__ import annotations

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy_config
from openpi.policies.libero_victr_retrieval import LiberoVictrBank
from openpi.policies.vfe_progress_client import VfeProgressClient
from openpi.serving import websocket_policy_server
from openpi.training import config


@dataclasses.dataclass
class Args:
    config: str = "pi05_victr_libero100_dino_progress"
    checkpoint_dir: str = tyro.MISSING
    corpus_dir: str = tyro.MISSING
    port: int = 8000
    vfe_host: str = "127.0.0.1"
    vfe_port: int | None = None


def main(args: Args) -> None:
    bank = LiberoVictrBank(args.corpus_dir)
    needs_progress = bank.metric in {"progress", "vision_progress"}
    bank.close()
    if needs_progress and args.vfe_port is None:
        raise ValueError("Progress-based VICTR rollout requires --vfe-port")
    progress_predictor = VfeProgressClient(args.vfe_host, args.vfe_port) if needs_progress else None
    policy = policy_config.create_trained_libero_victr_policy(
        config.get_config(args.config),
        args.checkpoint_dir,
        args.corpus_dir,
        progress_predictor=progress_predictor,
    )
    hostname = socket.gethostname()
    logging.info("Creating LIBERO VICTR server (host=%s, ip=%s)", hostname, socket.gethostbyname(hostname))
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
