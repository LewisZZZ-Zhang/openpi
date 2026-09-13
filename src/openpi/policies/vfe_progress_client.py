"""Client for VFE's localhost progress-prediction service."""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = []
    while length:
        chunk = sock.recv(length)
        if not chunk:
            raise EOFError("VFE progress server closed the connection")
        chunks.append(chunk)
        length -= len(chunk)
    return b"".join(chunks)


class VfeProgressClient:
    def __init__(self, host: str, port: int, timeout_seconds: float = 600.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_seconds = float(timeout_seconds)
        if not 0 < self.port <= 65535:
            raise ValueError(f"Invalid VFE progress server port: {self.port}")

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError(f"VFE request is too large: {len(payload)} bytes")
        with socket.create_connection((self.host, self.port), timeout=self.timeout_seconds) as sock:
            sock.settimeout(self.timeout_seconds)
            sock.sendall(struct.pack("!Q", len(payload)) + payload)
            size = struct.unpack("!Q", _recv_exact(sock, 8))[0]
            if size > MAX_MESSAGE_BYTES:
                raise ValueError(f"VFE response is too large: {size} bytes")
            response = pickle.loads(_recv_exact(sock, size))
        if not response.get("ok"):
            raise RuntimeError(f"VFE progress inference failed: {response.get('error', 'unknown error')}")
        return response

    def health(self) -> bool:
        return bool(self._request({"type": "health"})["ok"])

    def predict(self, observation: dict[str, Any], task_id: int) -> float:
        return float(
            self._request({"type": "predict", "task_id": int(task_id), "observation": observation})["progress"]
        )
