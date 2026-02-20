"""
Script to run the superres_app executable within a Docker container for inference (mode=2).
"""

import sys
import os
import argparse
import shlex
import logging

import docker_utils

import docker
from docker.types import DeviceRequest, Ulimit


logger = logging.getLogger(__name__)


def main():
    """main"""
    parser = argparse.ArgumentParser(
        description="Run superres_app in Docker (Inference Mode)"
    )
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file (INI)"
    )
    parser.add_argument("--image", default="trt-devel:latest", help="Docker image name")
    parser.add_argument(
        "--appdir",
        default="bin",
        help="Directory containing superres_app (default: bin)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    # Parse known args, leave others for the app
    args, unknown = parser.parse_known_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Find Project Root (git root or current dir's parent usually)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    config_abs_path = os.path.abspath(args.config)

    # Extract paths from config
    # We use default path_keys from docker_utils which covers plan-file, input-yuv-file, etc.
    files_to_check = docker_utils.get_config_paths(config_abs_path, project_root)
    # Also add config file itself if it's outside project root
    files_to_check.append(config_abs_path)

    # Resolve App Dir
    app_dir_arg = args.appdir
    host_app_dir = app_dir_arg

    if not os.path.isabs(app_dir_arg):
        cwd_path = os.path.abspath(app_dir_arg)
        proj_path = os.path.abspath(os.path.join(project_root, app_dir_arg))

        if os.path.exists(cwd_path):
            host_app_dir = cwd_path
        elif os.path.exists(proj_path):
            host_app_dir = proj_path
        else:
            host_app_dir = proj_path
    else:
        host_app_dir = app_dir_arg

    # Add app dir to mount check
    files_to_check.append(host_app_dir)

    logger.debug("Project Root: %s", project_root)
    logger.debug("Host App Dir: %s", host_app_dir)
    logger.debug("Found paths to check/mount: %s", files_to_check)

    # Prepare config path for inside container
    docker_config_path = args.config
    if config_abs_path.startswith(project_root):
        rel_path = os.path.relpath(config_abs_path, project_root)
        docker_config_path = rel_path
    else:
        docker_config_path = config_abs_path

    # Determine Container App Dir
    container_app_dir = ""
    if os.path.abspath(host_app_dir).startswith(project_root):
        rel = os.path.relpath(host_app_dir, project_root)
        container_app_dir = os.path.join("/workspace", rel)
    else:
        container_app_dir = host_app_dir

    app_executable = os.path.join(container_app_dir, "superres_app")

    # Construct App Command Parts
    app_cmd_parts = [app_executable, "--config", docker_config_path]
    app_cmd_parts.extend(unknown)

    # Convert to shell string to allow env var expansion
    app_cmd_str = " ".join(shlex.quote(s) for s in app_cmd_parts)
    # Prepend LD_LIBRARY_PATH setting
    final_cmd = f"LD_LIBRARY_PATH={container_app_dir}:$LD_LIBRARY_PATH {app_cmd_str}"

    logger.debug("Final Shell Command: %s", final_cmd)

    # Docker Client
    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        logger.error("Error connecting to Docker: %s", e)
        sys.exit(1)

    # GPU Config
    device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]

    # Ulimits
    ulimits = [
        Ulimit(name="memlock", soft=-1, hard=-1),
        Ulimit(name="stack", soft=67108864, hard=67108864),
    ]

    # Volumes
    volumes = docker_utils.get_mounts(project_root, files_to_check)

    # User
    user_id = os.getuid()
    group_id = os.getgid()

    logger.info("Executing superres_app (Inference) in Docker Image: %s...", args.image)

    try:
        container = client.containers.run(
            image=args.image,
            command=["/bin/bash", "-c", final_cmd],
            volumes=volumes,
            working_dir="/workspace",
            user=f"{user_id}:{group_id}",
            shm_size="8g",
            ulimits=ulimits,
            device_requests=device_requests,
            remove=True,
            detach=True,
            stdout=True,
            stderr=True,
        )

        # Stream logs
        for line in container.logs(stream=True):
            logger.info(line.decode("utf-8").strip())

    except docker.errors.ContainerError as e:
        logger.error("Container error: %s", e)
        sys.exit(e.exit_status)
    except docker.errors.ImageNotFound:
        logger.error("Image not found: %s", args.image)
        sys.exit(1)
    except docker.errors.APIError as e:
        logger.error("Docker API error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
