"""
Script to run SuperRes transcoding pipeline using Docker.

- Decodes input video using PyAV (ffmpeg) locally.
- Pipes raw YUV frames to `superres_app` running in a Docker container.
- Reads processed YUV frames from `superres_app` stdout.
- Encodes output video using PyAV (ffmpeg) locally.

Dependencies:
    - docker
    - av
    - pyyaml
"""

import sys
import os
import argparse
import threading
import logging
from pathlib import Path
from dataclasses import dataclass
from fractions import Fraction

import docker_utils

import yaml
import av
import docker
import numpy as np

# pylint: disable=broad-exception-caught

logger = logging.getLogger(__name__)

APP_TO_AV_FMT_MAP = {
    "i420": "yuv420p",
    "nv12": "nv12",
    "i422": "yuv422p",
    "nv16": "nv16",
    "i010": "yuv420p10le",
    "p010": "p010le",
    "i210": "yuv422p10le",
    "p210": "p210le",
}


@dataclass
class VideoMetadata:
    """
    Data class to store video stream metadata.
    """

    width: int
    height: int
    fps: Fraction
    pix_fmt: str
    app_input_fmt: str
    stream_index: int


def map_pix_fmt(av_fmt: str) -> str | None:
    """
    Map PyAV Pixel Format to SuperRes App Format
    superres_app supports:
        i420, nv12, i422, nv16,
        i010, p010, i210, p210,
        i016, p016, i216, p216
    """
    fmt_map = {
        "yuv420p": "i420",
        "nv12": "nv12",
        "yuv422p": "i422",
        "nv16": "nv16",
        "yuv420p10le": "i010",
        "p010le": "p010",
        "yuv422p10le": "i210",
        "p210le": "p210",
        "yuv420p16le": "i016",
        "p016le": "p016",
        "yuv422p16le": "i216",
        "p216le": "p216",
    }

    if av_fmt in fmt_map:
        return fmt_map[av_fmt]

    if av_fmt == "yuvj420p":
        return "i420"

    return None


def inspect_video(input_path: str) -> VideoMetadata:
    """
    Reads video file and extracts required metadata for the pipeline.
    """
    try:
        with av.open(input_path) as input_info:
            in_stream = input_info.streams.video[0]

            app_input_fmt = map_pix_fmt(in_stream.pix_fmt)
            if not app_input_fmt:
                logger.warning(
                    "Input pixel format '%s' not directly supported. Converting to i420.",
                    in_stream.pix_fmt,
                )
                app_input_fmt = "i420"

            metadata = VideoMetadata(
                width=in_stream.width,
                height=in_stream.height,
                fps=in_stream.average_rate,
                pix_fmt=in_stream.pix_fmt,
                app_input_fmt=app_input_fmt,
                stream_index=in_stream.index,
            )

            logger.info(
                "Input Video: %dx%d @ %.2f fps, format=%s -> %s",
                metadata.width,
                metadata.height,
                float(metadata.fps),
                metadata.pix_fmt,
                metadata.app_input_fmt,
            )
            return metadata

    except Exception as e:
        logger.error("Error reading input video info from '%s': %s", input_path, e)
        raise RuntimeError(f"Failed to inspect video: {input_path}") from e


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run SuperRes Transcoding pipeline via Docker."
    )
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--image", default="trt-devel:latest", help="Docker image name")
    parser.add_argument(
        "--app-dir",
        default="bin",
        help="Directory containing superres_app inside container (default: bin)",
    )
    parser.add_argument("-i", "--input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    # Optional arguments (override config)
    parser.add_argument("--width", type=int, help="Output Width (0=same as input)")
    parser.add_argument("--height", type=int, help="Output Height (0=same as input)")
    parser.add_argument("--prescale", type=float, help="Prescale factor (0.25-1.0)")
    parser.add_argument("--overlap", type=int, help="Overlap pixels (0-32)")
    parser.add_argument("--batches", type=int, help="Concurrent batches (1+)")
    parser.add_argument("--plan-file", help="Path to TensorRT plan file")

    return parser.parse_args()


def load_config(args) -> dict:
    """
    Load configuration from YAML file and override with CLI arguments.
    (Priority: CLI Arguments > YAML Config > Defaults)
    """
    # 1. Base Defaults
    final_config = {
        "input": None,
        "output": "output.mp4",
        "plan_file": None,
        "output_width": 0,
        "output_height": 0,
        "prescale": 1.0,
        "overlap": 4,
        "batches": 16,
    }

    config_path = Path(args.config) if getattr(args, "config", None) else None

    # 2. Load from YAML and Update
    if config_path and config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
                final_config.update(
                    {k: v for k, v in yaml_config.items() if k in final_config}
                )
            logger.info("Loaded config from %s", config_path)
        except yaml.YAMLError as e:
            logger.error("YAML parsing error: %s", e)
            raise
    elif config_path:
        logger.warning(
            "Config file %s not found. Using defaults/CLI args.", config_path
        )

    # 3. Override with CLI Arguments (Ignore None values)
    cli_mappings = {
        "input": getattr(args, "input", None),
        "output": getattr(args, "output", None),
        "plan_file": getattr(args, "plan_file", None),
        "output_width": getattr(args, "width", None),
        "output_height": getattr(args, "height", None),
        "prescale": getattr(args, "prescale", None),
        "overlap": getattr(args, "overlap", None),
        "batches": getattr(args, "batches", None),
    }

    for key, val in cli_mappings.items():
        if val is not None:
            final_config[key] = val

    # 4. Required Fields Validation
    if not final_config["input"]:
        raise ValueError("Input file needed (via --input or config file).")

    input_path = Path(final_config["input"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    if not final_config["plan_file"]:
        raise ValueError("Plan file needed (via --plan-file or config file).")

    # 5. Resolve Plan File Path gracefully using pathlib
    plan_path = Path(final_config["plan_file"])
    if not plan_path.is_absolute():
        # CWD에서 먼저 찾고, 없으면 Config 파일이 있는 디렉토리 기준으로 탐색
        if not plan_path.exists() and config_path:
            rel_to_config_dir = config_path.parent / plan_path
            if rel_to_config_dir.exists():
                plan_path = rel_to_config_dir

    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file '{plan_path}' not found.")

    final_config["plan_file"] = str(plan_path.resolve())

    return final_config


def setup_docker_container(
    client: docker.DockerClient,
    args,
    config: dict,
    video_meta: "VideoMetadata",
) -> docker.models.containers.Container:
    """
    Prepares volumes, environment, and CLI arguments, then creates the Docker container.
    """
    project_root = Path(__file__).resolve().parent.parent

    # 1. Mount Preparation
    files_to_mount = [config["plan_file"]]
    host_app_dir = Path(args.app_dir)

    if not host_app_dir.is_absolute():
        host_app_dir = project_root / host_app_dir

    if host_app_dir.exists():
        files_to_mount.append(str(host_app_dir))
        # Assuming docker_utils.get_mounts handles the heavy lifting
        volumes = docker_utils.get_mounts(str(project_root), files_to_mount)
        container_app_dir = _resolve_container_app_dir(str(host_app_dir), volumes)
    else:
        volumes = docker_utils.get_mounts(str(project_root), files_to_mount)
        container_app_dir = args.app_dir

    container_plan_file = config["plan_file"]
    if container_plan_file.startswith(str(project_root)):
        container_plan_file = (
            f"/workspace/{Path(container_plan_file).relative_to(project_root)}"
        )

    app_bin = f"{container_app_dir}/superres_app"

    # 2. Output Dimensions
    out_width = (
        config["output_width"] if config["output_width"] > 0 else video_meta.width
    )
    out_height = (
        config["output_height"] if config["output_height"] > 0 else video_meta.height
    )

    # 3. Build Command
    cmd_app = [
        app_bin,
        "--mode",
        "2",
        "--plan-file",
        container_plan_file,
        "--input-yuv-file",
        "-",  # 수정됨
        "--output-yuv-file",
        "-",  # 수정됨
        "--output-width",
        str(out_width),  # 언더바(_)에서 하이픈(-)으로 수정됨
        "--output-height",
        str(out_height),  # 언더바(_)에서 하이픈(-)으로 수정됨
        "--prescale",
        str(config["prescale"]),
        "--overlap",
        str(config["overlap"]),
        "--batches",
        str(config["batches"]),
        "--width",
        str(video_meta.width),
        "--height",
        str(video_meta.height),
        "--input-format",
        video_meta.app_input_fmt,  # 수정됨
    ]

    # 4. Create Container
    container = client.containers.create(
        image=args.image,
        command=cmd_app,
        volumes=volumes,
        working_dir="/workspace",
        user=f"{os.getuid()}:{os.getgid()}",
        environment={"LD_LIBRARY_PATH": f"{container_app_dir}:$LD_LIBRARY_PATH"},
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        ulimits=[
            docker.types.Ulimit(name="memlock", soft=-1, hard=-1),
            docker.types.Ulimit(name="stack", soft=67108864, hard=67108864),
        ],
        stdin_open=True,  # -i
        detach=True,  # -d
    )
    return container, out_width, out_height


def _resolve_container_app_dir(abs_host_app: str, volumes: dict) -> str:
    """Helper to find the best matching mount path for the app directory."""
    best_match_len = 0
    matched_bind = None

    for h_p, m_i in volumes.items():
        if abs_host_app.startswith(h_p):
            if len(h_p) > best_match_len:
                best_match_len = len(h_p)
                rel = os.path.relpath(abs_host_app, h_p)
                matched_bind = (
                    m_i["bind"] if rel == "." else os.path.join(m_i["bind"], rel)
                )

    return matched_bind if matched_bind else "/workspace"


def process_video_output(
    config: dict,
    docker_pipe: docker_utils.DockerDemuxer,
    video_meta: "VideoMetadata",
    out_width: int,
    out_height: int,
):
    """Reads raw frames from docker pipe and encodes them to output video."""

    fmt_map = {
        "i420": "yuv420p",
        "nv12": "nv12",
        "i422": "yuv422p",
        "nv16": "nv16",
        "i010": "yuv420p10le",
        "p010": "p010le",
        "i210": "yuv422p10le",
        "p210": "p210le",
    }
    av_out_fmt = fmt_map.get(video_meta.app_input_fmt, "yuv420p")

    y_size = out_width * out_height
    uv_size = (out_width // 2) * (out_height // 2)

    # Calculate Frame Size
    bpp_map = {
        "i422": 2.0,
        "nv16": 2.0,
        "i010": 3.0,
        "p010": 3.0,
        "i210": 4.0,
        "p210": 4.0,
    }
    bpp = bpp_map.get(video_meta.app_input_fmt, 1.5)
    frame_size = int(out_width * out_height * bpp)

    logger.info("Output Frame Size: %d bytes (Format: %s)", frame_size, av_out_fmt)

    with av.open(config["output"], "w") as output_container:
        out_stream = output_container.add_stream(
            "libx264", rate=video_meta.fps
        )  # 이전의 safe_fps 사용
        out_stream.options = {"preset": "ultrafast"}
        out_stream.width = out_width
        out_stream.height = out_height
        out_stream.pix_fmt = av_out_fmt

        while True:
            raw_data = docker_pipe.read(frame_size)
            if not raw_data:
                break
            if len(raw_data) < frame_size:
                logger.warning("Incomplete frame received (%d bytes).", len(raw_data))
                break

            frame = av.VideoFrame(out_width, out_height, av_out_fmt)

            if av_out_fmt == "yuv420p":
                frame.planes[0].update(raw_data[0:y_size])
                frame.planes[1].update(raw_data[y_size : y_size + uv_size])
                frame.planes[2].update(
                    raw_data[y_size + uv_size : y_size + uv_size * 2]
                )
            elif av_out_fmt == "nv12":
                frame.planes[0].update(raw_data[0:y_size])
                frame.planes[1].update(raw_data[y_size : y_size + uv_size * 2])

            for packet in out_stream.encode(frame):
                output_container.mux(packet)

        for packet in out_stream.encode():
            output_container.mux(packet)


def input_feeder(input_path: str, pipe_in, stream_index: int, target_format: str):
    """
    Decodes input video and writes raw frames to the pipe safely.
    """
    av_target_fmt = APP_TO_AV_FMT_MAP.get(target_format, "yuv420p")

    try:
        with av.open(input_path) as container:
            stream = container.streams[stream_index]

            for frame in container.decode(stream):
                if frame.format.name != av_target_fmt:
                    frame = frame.reformat(format=av_target_fmt)

                data = frame.to_ndarray(format=av_target_fmt).tobytes()

                try:
                    pipe_in.write(data)
                except BrokenPipeError:
                    break
                except Exception as e:
                    logger.error("Unexpected error writing to pipe: %s", e)
                    break

    except Exception as e:
        logger.exception("Critical unexpected error in input feeder: %s", e)
    finally:
        try:
            if hasattr(pipe_in, "shutdown_write"):
                pipe_in.shutdown_write()
        except Exception as e:
            logger.debug("Error while shutting down pipe_in write channel: %s", e)


def main():
    """Main Orchestrator for the Pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # 1. Initialization
        config = load_config(args)
        video_meta = inspect_video(config["input"])

        try:
            docker_client = docker.from_env()
        except docker.errors.DockerException as e:
            logger.error("Error connecting to Docker: %s", e)
            sys.exit(1)

        # 2. Setup and Start Container
        container, out_width, out_height = setup_docker_container(
            docker_client, args, config, video_meta
        )
        logger.info("Starting Docker Container...")

        sock = container.attach_socket(
            params={"stdin": 1, "stdout": 1, "stderr": 1, "stream": 1}
        )

        docker_pipe = docker_utils.DockerDemuxer(sock)
        container.start()

        # 3. Start I/O Threads
        feeder = threading.Thread(
            target=input_feeder,
            args=(
                config["input"],
                docker_pipe,
                video_meta.stream_index,
                video_meta.app_input_fmt,
            ),
            daemon=True,  # 데몬 스레드로 설정하여 메인 스레드 종료 시 같이 정리되도록 권장
        )
        feeder.start()

        # 4. Process Output (Main Thread Block)
        process_video_output(config, docker_pipe, video_meta, out_width, out_height)

    except Exception as e:
        logger.critical("Pipeline failed: %s", e)
        sys.exit(1)

    finally:
        # 5. Safe Cleanup
        if "container" in locals():
            try:
                container.kill()
                container.remove()
                logger.info("Container cleaned up.")
            except Exception as e:
                logger.debug("Container cleanup ignored or failed: %s", e)

        if "docker_pipe" in locals():
            docker_pipe.close()

        if "feeder" in locals() and feeder.is_alive():
            feeder.join(timeout=5.0)

        logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
