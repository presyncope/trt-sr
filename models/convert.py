import onnx
from onnxconverter_common import float16


srcfile = "xlsr-float.onnx/model.onnx"
dstfile = "xlsr-float.onnx/model_f16.onnx"


def convert_float_model(src, dst):
    model = onnx.load(src)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "n"

    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)

    onnx.save(model_fp16, dst)

    print("Model Output Shape:")
    for output in model.graph.output:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param
            for d in output.type.tensor_type.shape.dim
        ]
        print(f"  {output.name}: {shape}")

    print(f"Successfully converted {src} to {dst} (Dynamic Batch + FP16)")


convert_float_model(srcfile, dstfile)
