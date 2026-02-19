import argparse
import os
import sys
import subprocess
import threading
import av
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Real-ESRGAN video processing pipeline using PyAV and Pipes."
    )
    parser.add_argument("--input", help="Input video file")
    parser.add_argument("--output", help="Output video file")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml",
    )

    # Real-ESRGAN App specific arguments
    parser.add_argument(
        "--mode", type=int, default=2, help="Mode (1: build, 2: process, 3: both)"
    )
    parser.add_argument("--onnx-file", help="ONNX file (required for build mode)")
    parser.add_argument("--plan-file", help="Plan file")
    parser.add_argument(
        "--app-bin",
        default=str(Path().cwd().parent / "bin" / "real-esrgan-app"),
        help="Path to real-esrgan-app binary",
    )

    # New arguments matching main.cpp
    parser.add_argument(
        "--input-format",
        default="i420",
        help="Input Pixel Format (i420, nv12, i422, nv16, i010, p010, i210, p210, i016, p016, i216, p216)",
    )
    parser.add_argument(
        "--input-full-range", type=int, default=0, help="Input Full Range (0 or 1)"
    )
    parser.add_argument(
        "--output-width", type=int, default=0, help="Output Width (0=same as input)"
    )
    parser.add_argument(
        "--output-height", type=int, default=0, help="Output Height (0=same as input)"
    )
    parser.add_argument("--output-format", default="i420", help="Output Pixel Format")
    parser.add_argument(
        "--output-full-range", type=int, default=0, help="Output Full Range (0 or 1)"
    )
    parser.add_argument("--prescale", type=float, default=1.0, help="Prescale factor")
    parser.add_argument("--overlap", type=int, default=4, help="Overlap pixels")
    parser.add_argument(
        "--version-compatible",
        action="store_true",
        help="Enable version compatibility",
    )
    parser.add_argument(
        "--exclude-lean-runtime",
        action="store_true",
        help="Exclude lean runtime from plan",
    )
    # Batch and performance options
    parser.add_argument(
        "--batches", type=int, default=1, help="Concurrent batches (1-64)"
    )
    parser.add_argument(
        "--workspace", type=int, default=0, help="Max workspace size (bytes)"
    )
    parser.add_argument(
        "--min-batch", type=int, default=1, help="Min batch size for engine"
    )
    parser.add_argument(
        "--max-batch", type=int, default=8, help="Max batch size for engine"
    )
    parser.add_argument(
        "--opt-batch", type=int, default=8, help="Optimal batch size for engine"
    )
    parser.add_argument(
        "--strongly-typed",
        action="store_true",
        help="Enable strongly typed network",
    )

    # Parse config file first if provided or default exists
    # We use parse_known_args to find the config file path before full parsing
    temp_args, _ = parser.parse_known_args()
    config_path = temp_args.config

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config:
                # Convert keys to match argparse arguments (e.g. replace - with _)
                config = {k.replace("-", "_"): v for k, v in config.items()}
                parser.set_defaults(**config)

    args = parser.parse_args()

    # Manual validation for required arguments that might come from config
    if not args.input:
        parser.error("the following arguments are required: --input")
    if not args.output:
        parser.error("the following arguments are required: --output")
    if not args.plan_file:
        parser.error("the following arguments are required: --plan-file")

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    if not os.path.exists(args.app_bin):
        print(f"Error: App binary '{args.app_bin}' not found.")
        sys.exit(1)

    return args


def feed_process(input_path: str, process: subprocess.Popen, stream_index: int):
    """
    Reads frames from input video and writes raw YUV420P bytes to process.stdin.
    Running in a separate thread.
    """
    try:
        # Open separate container for thread safety
        container = av.open(input_path)
        stream = container.streams[stream_index]

        # Use efficient decoding
        # We need to ensure we output YUV420P
        for frame in container.decode(stream):
            # Reformat to yuv420p if necessary
            if frame.format.name != "yuv420p":
                frame = frame.reformat(format="yuv420p")

            # Write plane data
            # PyAV's to_ndarray().tobytes() is simplest, assuming contiguous memory
            # However, to_ndarray() creates a copy.
            # Direct plane access is better but Python `write` needs bytes.
            # Using to_ndarray for simplicity and robustness.
            # Note: real-esrgan expects planar YUV420: Y plane, then U plane, then V plane.

            # frame.to_ndarray() with format='yuv420p' returns a flat array of Y+U+V
            # This is exactly what we want.
            data = frame.to_ndarray(format="yuv420p").tobytes()
            process.stdin.write(data)

        process.stdin.flush()
    except Exception as e:
        print(f"Error in input feeder thread: {e}", file=sys.stderr)
    finally:
        # Close stdin to signal EOF to the app
        process.stdin.close()
        container.close()


def main():
    args = parse_args()

    # 1. Get Video Properties
    try:
        input_info = av.open(args.input)
        in_stream = input_info.streams.video[0]
        width = in_stream.width
        height = in_stream.height
        fps = in_stream.average_rate
        stream_index = in_stream.index
        input_info.close()

        print(f"Input Video: {width}x{height} @ {fps} fps")
    except Exception as e:
        print(f"Error reading input: {e}")
        sys.exit(1)

    # 2. Setup Real-ESRGAN App Process
    # We use /dev/stdin and /dev/stdout as files, and pipe actual data via subprocess
    cmd = [
        args.app_bin,
        "--mode",
        str(args.mode),
        "--plan-file",
        args.plan_file,
        "--input-yuv-file",
        "/dev/stdin",
        "--output-yuv-file",
        "/dev/stdout",
        "--width",
        str(width),
        "--height",
        str(height),
        "--input-format",
        args.input_format,
        "--input-full-range",
        str(args.input_full_range),
        "--output-format",
        args.output_format,
        "--output-full-range",
        str(args.output_full_range),
        "--prescale",
        str(args.prescale),
        "--overlap",
        str(args.overlap),
        "--batches",
        str(args.batches),
        "--workspace",
        str(args.workspace),
        "--min-batch",
        str(args.min_batch),
        "--max-batch",
        str(args.max_batch),
        "--opt-batch",
        str(args.opt_batch),
    ]
    if args.strongly_typed:
        cmd.append("--strongly-typed")
    if args.output_width > 0:
        cmd.extend(["--output-width", str(args.output_width)])
    if args.output_height > 0:
        cmd.extend(["--output-height", str(args.output_height)])

    if args.onnx_file:
        cmd.extend(["--onnx-file", args.onnx_file])

    if args.version_compatible:
        cmd.append("--version-compatible")

    if args.exclude_lean_runtime:
        cmd.append("--exclude-lean-runtime")

    print(f"Starting App: {' '.join(cmd)}")

    # Start process with pipes
    # stderr is inherited to see logs
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr
    )

    # 3. Start Input Feeder Thread
    feeder = threading.Thread(
        target=feed_process, args=(args.input, process, stream_index)
    )
    feeder.start()

    # 4. Process Output and Write to File
    try:
        output_container = av.open(args.output, "w")
        # Using h264_nvenc as requested/previously used.
        # If unavailable, user needs to change codec or install drivers.
        # Fallback handling could be added but let's stick to spec.
        # Per user request: Use libx264 with ultrafast preset for speed and compatibility
        out_stream = output_container.add_stream("libx264", rate=fps)
        out_stream.options = {"preset": "ultrafast"}

        out_width = args.output_width if args.output_width > 0 else width
        out_height = args.output_height if args.output_height > 0 else height

        out_stream.width = out_width
        out_stream.height = out_height
        out_stream.pix_fmt = "yuv420p"

        # Calculate frame size in bytes for YUV420P
        # Y=w*h, U=w*h/4, V=w*h/4 -> Total = w*h*1.5
        y_size = out_width * out_height
        uv_size = (out_width // 2) * (out_height // 2)
        frame_size = y_size + 2 * uv_size

        print("Processing...", flush=True)

        while True:
            # Read exact frame size
            # Note: stdout.read may return less if EOF or buffer issues,
            # essentially we need to ensure we get a full frame.
            raw_data = process.stdout.read(frame_size)

            if not raw_data:
                break

            if len(raw_data) < frame_size:
                # Unexpected EOF or partial frame
                print(
                    f"Warning: Incomplete frame read ({len(raw_data)} bytes). Stopping."
                )
                break

            # Create av.VideoFrame
            frame = av.VideoFrame(out_width, out_height, "yuv420p")

            # Copy data to planes
            # raw_data is flat YUV
            frame.planes[0].update(raw_data[0:y_size])
            frame.planes[1].update(raw_data[y_size : y_size + uv_size])
            frame.planes[2].update(raw_data[y_size + uv_size : y_size + uv_size * 2])

            # Encode
            for packet in out_stream.encode(frame):
                output_container.mux(packet)

        # Flush encoder
        for packet in out_stream.encode():
            output_container.mux(packet)

        output_container.close()

    except Exception as e:
        print(f"Error in output processing: {e}")
        process.terminate()
    finally:
        # Cleanup
        if process.poll() is None:
            process.terminate()
        feeder.join()
        print("Pipeline finished.")


if __name__ == "__main__":
    main()
