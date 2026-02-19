import argparse
import os
import sys
import subprocess
import threading
import yaml
import av
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SuperRes Runtime-Only (Inference) pipeline."
    )
    parser.add_argument(
        "--config", default="superres-config.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--app-dir",
        help="Directory containing superres_app binary (default: ../bin relative to script)",
    )
    parser.add_argument("-i", "--input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")

    # Optional arguments (override config)
    parser.add_argument("--width", type=int, help="Output Width (0=same as input)")
    parser.add_argument("--height", type=int, help="Output Height (0=same as input)")
    parser.add_argument("--prescale", type=float, help="Prescale factor (0.25-1.0)")
    parser.add_argument("--overlap", type=int, help="Overlap pixels (0-32)")
    parser.add_argument("--batches", type=int, help="Concurrent batches (1+)")
    parser.add_argument("--plan-file", help="Path to TensorRT plan file")

    return parser.parse_args()


def load_config(args):
    config = {}

    # 1. Load from YAML if exists
    config_path = args.config
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            print(f"Loaded config from {config_path}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    else:
        print(f"Config file {config_path} not found. Using defaults/CLI args.")

    # 2. Key mapping (CLI arg name -> Config key)
    # Using simple priority: CLI > Config > Default

    def get_val(cli_val, config_key, default_val=None):
        if cli_val is not None:
            return cli_val
        if config_key in config:
            return config[config_key]
        return default_val

    final_config = {}

    final_config["input"] = get_val(args.input, "input")
    final_config["output"] = get_val(args.output, "output", "output.mp4")

    # App Dir parsing
    default_app_dir = str(Path(__file__).parent.parent / "bin")
    final_config["app_dir"] = get_val(args.app_dir, "app_dir", default_app_dir)
    final_config["app_bin"] = os.path.join(final_config["app_dir"], "superres_app")

    final_config["plan_file"] = get_val(args.plan_file, "plan_file")

    final_config["output_width"] = int(get_val(args.width, "output_width", 0))
    final_config["output_height"] = int(get_val(args.height, "output_height", 0))
    final_config["prescale"] = float(get_val(args.prescale, "prescale", 1.0))
    final_config["overlap"] = int(get_val(args.overlap, "overlap", 4))
    final_config["batches"] = int(get_val(args.batches, "batches", 16))

    # Required check
    if not final_config["input"]:
        print("Error: Input file needed (via --input or config file).")
        sys.exit(1)

    if not os.path.exists(final_config["input"]):
        print(f"Error: Input file '{final_config['input']}' not found.")
        sys.exit(1)

    if not final_config["plan_file"]:
        print("Error: Plan file needed (via --plan-file or config file).")
        sys.exit(1)

    # Resolve plan file path
    # If relative, make it relative to the script or cwd?
    # Usually easier to keep as provided, but let's check existence
    if not os.path.exists(final_config["plan_file"]):
        # Try resolving relative to config file if loaded?
        # For simplicity, assume path is relative to CWD or absolute
        pass

    if not os.path.exists(final_config["app_bin"]):
        print(f"Error: App binary '{final_config['app_bin']}' not found.")
        print(f"Please build the app or specify correct --app-dir.")
        sys.exit(1)

    return final_config


def map_pix_fmt(av_fmt):
    """
    Map PyAV Pixel Format to SuperRes App Format
    superres_app supports: i420, nv12, i422, nv16, i010, p010, i210, p210, i016, p016, i216, p216
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
        # Add more if needed
    }

    if av_fmt in fmt_map:
        return fmt_map[av_fmt]

    # Fallback or specific handling for common unsupported formats
    if av_fmt == "yuvj420p":
        return "i420"  # JPEG range, but logic handles full range flag separately

    return None


def feed_process(
    input_path: str, process: subprocess.Popen, stream_index: int, target_format: str
):
    """
    Reads frames from input video and writes raw bytes to process.stdin.
    """
    try:
        container = av.open(input_path)
        stream = container.streams[stream_index]

        # PyAV format string for reformatting if needed
        # We need a reverse map or just hardcode for common target_formats
        # Since map_pix_fmt maps av->app, we need app->av to force reformat

        av_target_fmt = None
        # Simple reverse lookup (inefficient but small map)
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
        for k, v in fmt_map.items():
            if v == target_format:
                av_target_fmt = k
                break

        if not av_target_fmt:
            # Fallback to yuv420p if unknown
            av_target_fmt = "yuv420p"

        for frame in container.decode(stream):
            if frame.format.name != av_target_fmt:
                frame = frame.reformat(format=av_target_fmt)

            # to_ndarray().tobytes() creates a copy but is safe
            data = frame.to_ndarray(format=av_target_fmt).tobytes()
            process.stdin.write(data)

        process.stdin.flush()
    except Exception as e:
        print(f"Error in input feeder thread: {e}", file=sys.stderr)
    finally:
        process.stdin.close()
        container.close()


def main():
    args = parse_args()
    config = load_config(args)

    # 1. Inspect Input Video
    try:
        input_info = av.open(config["input"])
        in_stream = input_info.streams.video[0]

        width = in_stream.width
        height = in_stream.height
        fps = in_stream.average_rate
        pix_fmt = in_stream.pix_fmt
        stream_index = in_stream.index

        # Determine Input Format for App
        app_input_fmt = map_pix_fmt(pix_fmt)
        if not app_input_fmt:
            print(
                f"Warning: Input pixel format '{pix_fmt}' not directly supported. Converting to i420."
            )
            app_input_fmt = "i420"

        print(
            f"Input Video: {width}x{height} @ {float(fps):.2f} fps, format={pix_fmt} -> {app_input_fmt}"
        )
        input_info.close()

    except Exception as e:
        print(f"Error reading input video info: {e}")
        sys.exit(1)

    # 2. Setup SuperRes App Process

    # Determine Output Dimensions
    out_width = config["output_width"] if config["output_width"] > 0 else width
    out_height = config["output_height"] if config["output_height"] > 0 else height

    env = os.environ.copy()
    # Add app_dir to LD_LIBRARY_PATH to find libsuperres.so
    app_dir = config["app_dir"]
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = f"{app_dir}:{env['LD_LIBRARY_PATH']}"
    else:
        env["LD_LIBRARY_PATH"] = app_dir

    cmd = [
        config["app_bin"],
        "--mode",
        "2",  # Process Only
        "--plan-file",
        config["plan_file"],
        "--input-yuv-file",
        "/dev/stdin",
        "--output-yuv-file",
        "/dev/stdout",
        "--width",
        str(width),
        "--height",
        str(height),
        "--input-format",
        app_input_fmt,
        "--input-full-range",
        "1",  # Fixed as per request
        "--output-width",
        str(out_width),
        "--output-height",
        str(out_height),
        "--output-format",
        app_input_fmt,  # Same as input
        "--output-full-range",
        "1",  # Fixed
        "--prescale",
        str(config["prescale"]),
        "--overlap",
        str(config["overlap"]),
        "--batches",
        str(config["batches"]),
    ]

    print(f"Starting App: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, env=env
    )

    # 3. Start Feeder Thread
    feeder = threading.Thread(
        target=feed_process,
        args=(config["input"], process, stream_index, app_input_fmt),
    )
    feeder.start()

    # 4. Process Output
    try:
        output_container = av.open(config["output"], "w")

        # Use libx264 with ultrafast preset
        out_stream = output_container.add_stream("libx264", rate=fps)
        out_stream.options = {"preset": "ultrafast"}

        out_stream.width = out_width
        out_stream.height = out_height

        # We need to map app_input_fmt back to av format for the encoder
        # This assumes output format == input format (which we set in CLI args)
        av_out_fmt = "yuv420p"  # Default fallback
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
        if app_input_fmt in fmt_map:
            av_out_fmt = fmt_map[app_input_fmt]

        out_stream.pix_fmt = av_out_fmt

        # Calculate frame size
        # This is tricky for different formats.
        # Approximate size logic or use PyAV to calculate?
        # PyAV doesn't have a static function for size.
        # We implement basic size calc for supported formats.

        bpp = 1.0  # bytes per pixel (luma + chroma)
        if app_input_fmt in ["i420", "nv12"]:
            bpp = 1.5
        elif app_input_fmt in ["i422", "nv16"]:
            bpp = 2.0
        elif app_input_fmt in ["i010", "p010"]:
            bpp = 3.0  # 10 bit often 16bit storage
        elif app_input_fmt in ["i210", "p210"]:
            bpp = 4.0
        # ... verify 10/16 bit storage. usually 2 bytes per component.
        # 10bit: 4:2:0 -> 1.5 * 2 = 3 bytes

        # For simplicity, if not standard 8-bit 420/422, we might strictly check or warn.
        # Let's support 8-bit YUV420 for now as primary goal.

        if app_input_fmt not in ["i420", "nv12"]:
            print(
                f"Warning: Output frame size calculation for {app_input_fmt} might be inaccurate."
            )
            # Fallback manual calcs:
            if "p10" in av_out_fmt or "p16" in av_out_fmt:
                # 10/16 bit use 2 bytes per sample
                if "420" in av_out_fmt or "p0" in app_input_fmt:
                    bpp = 3.0
                elif "422" in av_out_fmt or "p2" in app_input_fmt:
                    bpp = 4.0

        frame_size = int(out_width * out_height * bpp)
        print(f"Output Frame Size: {frame_size} bytes (Format: {av_out_fmt})")

        while True:
            raw_data = process.stdout.read(frame_size)
            if not raw_data:
                break

            if len(raw_data) < frame_size:
                print(f"Warning: Incomplete frame ({len(raw_data)} bytes). Stopping.")
                break

            frame = av.VideoFrame(out_width, out_height, av_out_fmt)

            # Populate frame data from raw bytes
            # PyAV provides logic to wrap buffers but `frame.planes[i].update()` is safer
            # We must know the layout.

            # Simple case: YUV420P
            if av_out_fmt == "yuv420p":
                y_sz = out_width * out_height
                uv_sz = (out_width // 2) * (out_height // 2)
                frame.planes[0].update(raw_data[0:y_sz])
                frame.planes[1].update(raw_data[y_sz : y_sz + uv_sz])
                frame.planes[2].update(raw_data[y_sz + uv_sz :])
            elif av_out_fmt == "nv12":
                y_sz = out_width * out_height
                uv_sz = (
                    out_width // 2
                ) * out_height  # Interleaved UV, 1/2 width * height? No.
                # NV12: Y plane (w*h), then UV plane (w*h/2)
                uv_sz = out_width * (out_height // 2)
                frame.planes[0].update(raw_data[0:y_sz])
                frame.planes[1].update(raw_data[y_sz:])
            else:
                # Generic fallback (might fail if strides differ)
                # Converting bytes to ndarray via numpy might be needed
                # For now, if complex formats are used, we might need a more robust reader
                # or simply force i420 for output.
                print(
                    f"Error: Output writing for format {av_out_fmt} not fully implemented in script."
                )
                process.terminate()
                sys.exit(1)

            for packet in out_stream.encode(frame):
                output_container.mux(packet)

        # Flush
        for packet in out_stream.encode():
            output_container.mux(packet)

        output_container.close()

    except Exception as e:
        print(f"Error in output processing: {e}")
    finally:
        if process.poll() is None:
            process.terminate()
        feeder.join()
        print("Pipeline finished.")


if __name__ == "__main__":
    main()
