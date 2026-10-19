"""One-time ONNX graph fixes so downloaded models run fully on the GPU."""
import os

import onnx
from onnx import helper, numpy_helper
import numpy as np


def gpu_friendly_conv_transpose(src_path: str, dst_path: str) -> str:
    """
    onnxruntime's CUDA provider does not support ConvTranspose with asymmetric padding and runs
    those nodes on the CPU (XSeg has six: pads [0, 0, 1, 1]). Rewrites each one as an unpadded
    ConvTranspose followed by a Slice that trims the padding, which is mathematically identical.
    Returns dst_path (written once, reused afterwards).
    """
    if os.path.exists(dst_path) and os.path.getmtime(dst_path) >= os.path.getmtime(src_path):
        return dst_path
    model = onnx.load(src_path)
    nodes = []
    for node in model.graph.node:
        pads = next((list(helper.get_attribute_value(a)) for a in node.attribute if a.name == 'pads'), None)
        if node.op_type != 'ConvTranspose' or not pads or pads[:len(pads) // 2] == pads[len(pads) // 2:]:
            nodes.append(node)
            continue
        spatial = len(pads) // 2
        begin, end = pads[:spatial], pads[spatial:]
        for attribute in node.attribute:
            if attribute.name == 'pads':
                attribute.ints[:] = [0] * len(pads)
        unpadded = node.output[0] + '_unpadded'
        trimmed = node.output[0]
        node.output[0] = unpadded
        prefix = node.name or unpadded
        constants = {
            'starts': np.array(begin, dtype=np.int64),
            'ends': np.array([-e if e else np.iinfo(np.int64).max for e in end], dtype=np.int64),
            'axes': np.arange(2, 2 + spatial, dtype=np.int64),
        }
        for key, value in constants.items():
            model.graph.initializer.append(numpy_helper.from_array(value, f'{prefix}_trim_{key}'))
        nodes.append(node)
        nodes.append(helper.make_node('Slice', [unpadded] + [f'{prefix}_trim_{k}' for k in ('starts', 'ends', 'axes')],
                                      [trimmed], name=f'{prefix}_trim'))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, dst_path)
    return dst_path
