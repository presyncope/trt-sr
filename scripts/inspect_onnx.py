import argparse
import onnx
import onnx.numpy_helper
import sys
from onnx import shape_inference

def get_dtype_size(onnx_dtype):
    mapping = {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT16: 2,
        onnx.TensorProto.INT16: 2,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.STRING: 1, 
        onnx.TensorProto.BOOL: 1,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.UINT32: 4,
        onnx.TensorProto.UINT64: 8,
    }
    return mapping.get(onnx_dtype, 4)

def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0 : 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power and n < 4:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def inspect_onnx_model(model_path):
    print(f"Loading model: {model_path}")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Try to infer shapes if they are missing
    try:
        print("Running shape inference...")
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed. Some sizes might be missing. {e}")

    graph = model.graph
    
    print(f"\nModel Graph Name: {graph.name}")
    print("=" * 60)

    # Inputs
    print(f"Inputs ({len(graph.input)}):")
    for inp in graph.input:
        tensor_type = inp.type.tensor_type
        shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tensor_type.shape.dim]
        dtype = tensor_type.elem_type
        print(f"  Name: {inp.name}")
        print(f"  Type: {onnx.TensorProto.DataType.Name(dtype)}")
        print(f"  Shape: {shape}")
        print("-" * 20)

    # Outputs
    print(f"\nOutputs ({len(graph.output)}):")
    for out in graph.output:
        tensor_type = out.type.tensor_type
        shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tensor_type.shape.dim]
        dtype = tensor_type.elem_type
        print(f"  Name: {out.name}")
        print(f"  Type: {onnx.TensorProto.DataType.Name(dtype)}")
        print(f"  Shape: {shape}")
        print("-" * 20)

    # Initializers (Weights)
    print(f"\nInitializers/Weights ({len(graph.initializer)}):")
    total_weight_size = 0
    for init in graph.initializer:
        size = 1
        for dim in init.dims:
            size *= dim
        byte_size = size * get_dtype_size(init.data_type)
        total_weight_size += byte_size
        print(f"  Name: {init.name}")
        print(f"  Shape: {list(init.dims)}")
        print(f"  Size: {format_bytes(byte_size)}")
    
    print(f"\nTotal Weights Memory: {format_bytes(total_weight_size)}")

    # Intermediate Tensors (Value Info)
    print(f"\nIntermediate Tensors ({len(graph.value_info)}):")
    total_activation_size = 0
    for vi in graph.value_info:
        tensor_type = vi.type.tensor_type
        shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tensor_type.shape.dim]
        
        # Calculate size if shape is fully known
        is_dynamic = False
        elem_count = 1
        for s in shape:
            if isinstance(s, int):
                elem_count *= s
            else:
                is_dynamic = True
                break
        
        size_str = "Dynamic"
        if not is_dynamic:
            byte_size = elem_count * get_dtype_size(tensor_type.elem_type)
            total_activation_size += byte_size
            size_str = format_bytes(byte_size)
            
        print(f"  Name: {vi.name}")
        print(f"  Shape: {shape}")
        print(f"  Size: {size_str}")

    if total_activation_size > 0:
        print(f"\nTotal Estimated Intermediate Tensor Memory (for one pass, if shapes known): {format_bytes(total_activation_size)}")
    else:
        print("\nCould not estimate intermediate memory (dynamic shapes or missing inference).")
        
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ONNX model structure and memory usage.")
    parser.add_argument("model_path", help="Path to the ONNX model file")
    args = parser.parse_args()
    inspect_onnx_model(args.model_path)
