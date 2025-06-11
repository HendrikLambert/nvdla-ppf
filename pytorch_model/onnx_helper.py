import torch
import onnx
from onnx import numpy_helper, helper
from modules.pfb_module import PFBModule

def export_model(model: PFBModule, onnx_file: str, onnx_version: int = 17):
    example_inputs = torch.zeros(model.batch_size, 2 * model.P, 1, model.M)
    example_inputs[0, 0, 0, 0] = 1.0
    # example_inputs[0, 0, 0, 1] = 2.0

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        example_inputs,
        onnx_file + f"-{model.P}-{model.M}-{model.batch_size}.onnx",
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        opset_version=onnx_version,
        #   dynamic_axes={
        #     'input': {0: 'batch_size'},
        #     'output': {0: 'batch_size'}
        #   }
    )


def extract_constant_value(node):
    """Extract the value from a Constant node."""
    for attr in node.attribute:
        if attr.name == "value":
            return numpy_helper.to_array(attr.t)
    raise ValueError(f"No 'value' attribute found in Constant node: {node.name}")


def convert_slice_to_opset1(model):
    graph = model.graph

    # Collect initializers and Constant nodes as name → value
    const_values = {}
    node_map = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}
    for output_name, node in node_map.items():
        const_values[output_name] = extract_constant_value(node)

    new_nodes = []
    used_const_inputs = set()

    for node in graph.node:
        if node.op_type == "Slice" and len(node.input) >= 3:
            data_input = node.input[0]
            starts_input = node.input[1]
            ends_input = node.input[2]
            axes_input = node.input[3] if len(node.input) > 3 else None
            steps = node.input[4] if len(node.input) > 4 else None

            if all(i in const_values for i in [starts_input, ends_input]) and (
                axes_input is None or axes_input in const_values
            ):
                starts = const_values[starts_input].tolist()
                ends = const_values[ends_input].tolist()
                axes = (
                    const_values[axes_input].tolist()
                    if axes_input
                    else list(range(len(starts)))
                )

                # Create a Slice (v1) node using attributes
                new_slice = helper.make_node(
                    "Slice",
                    inputs=[data_input],
                    outputs=node.output,
                    starts=starts,
                    ends=ends,
                    axes=axes,
                )
                new_nodes.append(new_slice)
                used_const_inputs.update([starts_input, ends_input])
                if axes_input:
                    used_const_inputs.add(axes_input)
                if steps:
                    used_const_inputs.add(steps)
            else:
                raise ValueError(
                    f"Cannot convert Slice node {node.name} — slicing inputs not constant."
                )
        else:
            new_nodes.append(node)

    # Remove used Constant nodes
    new_nodes = [
        n
        for n in new_nodes
        if not (n.op_type == "Constant" and n.output[0] in used_const_inputs)
    ]
    # new_nodes = [n for n in new_nodes if not (n.op_type == 'Constant')]

    # Replace graph nodes
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    # Replace opset with v1
    model.ClearField("opset_import")
    model.opset_import.extend([onnx.helper.make_opsetid("", 13)])
    print("opset", model.opset_import)

    return model


def post_process_onnx(onnx_file: str):
    model = onnx.load(onnx_file)

    # Convert Slice nodes from v10 to v1
    model = convert_slice_to_opset1(model)

    onnx.save(model, onnx_file)