"""
Shared utilities for Docker execution scripts.
"""

import os
import configparser
import logging
import socket
import struct

__all__ = [
    "get_abs_path",
    "get_config_paths",
    "get_mounts",
    "DockerDemuxer",
]

logger = logging.getLogger(__name__)


def get_abs_path(path: str, base_dir: str) -> str:
    """get absolute path of a file"""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def get_config_paths(
    config_path: str, project_root: str, path_keys: list[str] = None
) -> list[str]:
    """
    Parses the INI config file and extracts relevant paths.
    Returns a list of absolute paths found in the config.
    """
    if not os.path.exists(config_path):
        return []

    if path_keys is None:
        path_keys = ["plan-file", "onnx-file", "input-yuv-file", "output-yuv-file"]

    config_dir = os.path.dirname(os.path.abspath(config_path))

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    config = configparser.ConfigParser()
    try:
        config.read_string(f"[dummy]\n{content}")
    except configparser.Error as e:
        logger.warning("Failed to parse config file %s: %s", config_path, e)
        return []

    # "dummy" 섹션이 없으면 빈 딕셔너리를 반환하도록 하여 if 문 제거
    section = config["dummy"] if "dummy" in config else {}
    paths = []

    for key in path_keys:
        val = section.get(key)

        # Guard Clause: 값이 없거나 공백이면 즉시 다음 루프로 이동
        if not val or not val.strip():
            continue

        clean_val = val.strip()

        # 1. project_root 기준 경로 확인
        abs_path_root = get_abs_path(clean_val, project_root)
        if os.path.exists(abs_path_root):
            paths.append(abs_path_root)
            continue  # 찾았으면 하위 로직(config_dir 확인)을 건너뜀

        # 2. config_dir 기준 경로 확인 (else 블록 제거)
        abs_path_cfg = get_abs_path(clean_val, config_dir)
        if os.path.exists(abs_path_cfg):
            paths.append(abs_path_cfg)

    return paths


def get_mounts(project_root, extra_paths):
    """
    Generates Docker volume mounts dictionary.
    Mounts project_root to /workspace.
    Mounts extra_paths using mirror mounting (host path == container path).
    Returns: dict {host_path: {'bind': container_path, 'mode': 'rw'}}
    """
    volumes = {}

    # Mount Project Root
    volumes[project_root] = {"bind": "/workspace", "mode": "rw"}

    # Mount extra paths
    mounted_dirs = set()

    for path in extra_paths:
        # Resolve absolute path
        abs_path = os.path.abspath(path)

        # If path is inside project root, it's already covered
        if abs_path.startswith(project_root):
            continue

        # If strictly outside, mount the parent directory
        parent_dir = os.path.dirname(abs_path)

        # Check if we already mounted this dir or a parent of it
        already_mounted = False
        for m_dir in mounted_dirs:
            if parent_dir.startswith(m_dir):
                already_mounted = True
                break

        if already_mounted:
            continue

        volumes[parent_dir] = {"bind": parent_dir, "mode": "rw"}
        mounted_dirs.add(parent_dir)

    return volumes


class DockerDemuxer:
    """
    Simple and fast wrapper for Docker raw stream.
    Reads/Writes pure raw bytes directly to the container's standard I/O.
    """

    def __init__(self, sock):
        self.sock = sock
        self._raw_sock = getattr(sock, "_sock", sock)

        if hasattr(self._raw_sock, "setblocking"):
            self._raw_sock.setblocking(True)

    def write(self, data: bytes):
        """Write raw data to the stdin of the Docker container."""
        try:
            if hasattr(self._raw_sock, "sendall"):
                self._raw_sock.sendall(data)
            elif hasattr(self.sock, "write"):
                self.sock.write(data)
            else:
                raise RuntimeError("Provided socket does not support writing.")
        except (OSError, BrokenPipeError) as e:
            logger.error("Pipe connection lost while writing to container: %s", e)
            raise

    def read(self, n: int) -> bytes | None:
        """
        Read exactly n bytes of raw payload from the stdout stream.
        No header demultiplexing, just pure byte aggregation.
        """
        data = bytearray()
        while len(data) < n:
            to_read = n - len(data)
            try:
                if hasattr(self._raw_sock, "recv"):
                    chunk = self._raw_sock.recv(to_read)
                elif hasattr(self.sock, "read"):
                    chunk = self.sock.read(to_read)
                else:
                    break

                if not chunk:  # Socket gracefully closed or EOF
                    break
                data.extend(chunk)
            except OSError as e:
                logger.error("Socket IO Error during read: %s", e)
                break

        # 요청한 바이트를 다 못 채웠으면 불완전한 데이터로 간주하고 None 반환
        return bytes(data) if len(data) == n else None

    def close(self):
        """Gracefully close the underlying socket."""
        try:
            if hasattr(self.sock, "close"):
                self.sock.close()
        except Exception:
            pass

    def shutdown_write(self):
        """
        Signals EOF to the container's STDIN by half-closing the socket,
        allowing STDOUT to remain open for reading the remaining frames.
        """
        try:
            if hasattr(self._raw_sock, "shutdown"):
                self._raw_sock.shutdown(socket.SHUT_WR)
                logger.debug("Sent EOF to Docker STDIN (Write channel closed).")
        except Exception as e:
            logger.debug("Failed to shutdown write channel: %s", e)
