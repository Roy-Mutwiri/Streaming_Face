"""
TensorRT engines built directly with the TensorRT API, for models onnxruntime's TensorRT
provider handles badly.

inswapper_128 ('stats' policy): its instance-norm layers square activations of up to ~1500, so
the variance reaches millions and overflows FP16 (max 65504). Plain FP16 (inswapper_128_fp16.onnx,
or TensorRT's FP16 mode) produces smeared, blocky faces. Convolutions and matrix multiplies run in
FP16 (where the speed is) and the norm statistics are pinned to FP32: FP32 output at FP16 speed.

Mask models ('fp32' policy): XSeg's normalisation layers make TensorRT's FP16 build crash, and
the parser gains little from FP16, so both are built in FP32/TF32.

Engines are built once in a child process (TensorRT can terminate the whole process on some
graphs) and cached in models/trt_cache. If a build fails, callers fall back to onnxruntime.
"""
import hashlib
import os
import subprocess
import sys
import threading
from typing import Dict, List, Optional

import numpy as np

from modules.accel import TRT_CACHE_DIR, _tensorrt_available

FP16_LAYER_TYPES = ('CONVOLUTION', 'DECONVOLUTION', 'MATRIX_MULTIPLY', 'FULLY_CONNECTED')
LOGGER = None  # TensorRT keeps one global logger; creating more just triggers a warning
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _logger():
    global LOGGER
    import tensorrt as trt
    if LOGGER is None:
        LOGGER = trt.Logger(trt.Logger.WARNING)
    return LOGGER


def engine_path(onnx_path: str, pin_policy: str) -> str:
    import tensorrt as trt
    import torch
    stat = os.stat(onnx_path)
    key = f'{os.path.basename(onnx_path)}|{stat.st_size}|{trt.__version__}|{torch.cuda.get_device_name(0)}|{pin_policy}|mixed-v3'
    os.makedirs(TRT_CACHE_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    return os.path.join(TRT_CACHE_DIR, f'{stem}_mixed_{hashlib.md5(key.encode()).hexdigest()[:16]}.engine')


def _stats_layers(network) -> set:
    """
    Indices of the layers that compute normalisation statistics: every REDUCE and UNARY (sqrt/rsqrt)
    layer, the elementwise layers feeding them (the squaring) and the elementwise layers that take
    their result (the "+eps"). These overflow FP16 in inswapper's instance norm.
    """
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    producer = {}
    for i, layer in enumerate(layers):
        for j in range(layer.num_outputs):
            producer[layer.get_output(j).name] = i
    stats = {i for i, layer in enumerate(layers) if layer.type.name in ('REDUCE', 'UNARY')}
    for i in list(stats):
        for k in range(layers[i].num_inputs):
            tensor = layers[i].get_input(k)
            p = producer.get(tensor.name) if tensor is not None else None
            if p is not None and layers[p].type.name == 'ELEMENTWISE':
                stats.add(p)
    for i, layer in enumerate(layers):
        if layer.type.name == 'ELEMENTWISE':
            inputs = [layer.get_input(k) for k in range(layer.num_inputs)]
            if any(t is not None and producer.get(t.name) in stats and layers[producer[t.name]].type.name == 'REDUCE' for t in inputs):
                stats.add(i)
    return stats


def build_engine(onnx_path: str, pin_policy: str) -> bytes:
    """pin_policy: 'stats' (FP16 + FP32 norm statistics), 'plain' (FP16) or 'fp32' (no FP16)."""
    import tensorrt as trt
    logger = _logger()
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError('TensorRT could not parse ' + onnx_path)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if pin_policy != 'fp32':
        config.set_flag(trt.BuilderFlag.FP16)
    # Dynamic (batch) dimensions get a fixed profile of 1
    dynamic = [network.get_input(i) for i in range(network.num_inputs) if -1 in tuple(network.get_input(i).shape)]
    if dynamic:
        profile = builder.create_optimization_profile()
        for tensor in dynamic:
            shape = tuple(max(d, 1) for d in tensor.shape)
            profile.set_shape(tensor.name, shape, shape, shape)
        config.add_optimization_profile(profile)
    if pin_policy == 'stats':
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        stats = _stats_layers(network)
        float_types = (trt.float32, trt.float16)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if (i in stats and layer.type.name not in FP16_LAYER_TYPES and layer.type.name != 'CONSTANT'
                    and all(layer.get_output(j).dtype in float_types for j in range(layer.num_outputs))):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('TensorRT engine build failed')
    return bytes(engine)


class TRTSession:
    """Minimal drop-in for the parts of onnxruntime.InferenceSession the app uses."""

    def __init__(self, path: str, pin_policy: str):
        import tensorrt as trt
        import torch
        self.trt, self.torch = trt, torch
        self.lock = threading.Lock()
        with open(path, 'rb') as f:
            self.engine = trt.Runtime(_logger()).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('could not load ' + path)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.buffers: Dict[str, 'torch.Tensor'] = {}
        # Dynamic (batch) dimensions are fixed to 1
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, tuple(max(d, 1) for d in self.engine.get_tensor_shape(name)))
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = torch.float32 if self.engine.get_tensor_dtype(name) == trt.float32 else torch.float16
            self.buffers[name] = torch.empty(shape, dtype=dtype, device='cuda')
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            (self.input_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.output_names).append(name)
        self.pin_policy = pin_policy

    # --- onnxruntime.InferenceSession-like API ---
    def get_inputs(self):
        return [_IO(n, list(self.buffers[n].shape)) for n in self.input_names]

    def get_outputs(self):
        return [_IO(n, list(self.buffers[n].shape)) for n in self.output_names]

    def get_providers(self):
        return [f'TensorRT ({self.pin_policy})']

    def run(self, output_names, feeds):
        return [t.float().cpu().numpy() for t in self.run_torch(feeds, output_names)]

    def run_torch(self, feeds, output_names=None, postprocess=None):
        """
        Runs the engine and returns the outputs as GPU tensors. `postprocess` (tensors -> result)
        runs on the GPU under the same lock/stream, so large outputs never have to reach the CPU.
        """
        torch = self.torch
        with self.lock, torch.cuda.stream(self.stream):
            for name, value in feeds.items():
                self.buffers[name].copy_(torch.from_numpy(np.ascontiguousarray(value)))
            self.context.execute_async_v3(self.stream.cuda_stream)
            outputs = [self.buffers[name] for name in (output_names or self.output_names)]
            result = postprocess(outputs) if postprocess else [o.clone() for o in outputs]
            self.stream.synchronize()
        return result


class _IO:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


BUILD_LOCK = threading.Lock()


def create_session(onnx_path: str, pin_policy: str) -> Optional[TRTSession]:
    """
    Returns a TensorRT session, or None if TensorRT is unavailable or the engine cannot be built
    (callers then fall back to onnxruntime). Missing engines are built in a child process.
    """
    if not _tensorrt_available():
        return None
    try:
        path = engine_path(onnx_path, pin_policy)
        with BUILD_LOCK:
            if not os.path.exists(path):
                print(f'[TRT] Building engine for {os.path.basename(onnx_path)} (one time, a few minutes)...', flush=True)
                result = subprocess.run([sys.executable, '-m', 'modules.trt_engine', onnx_path, pin_policy], cwd=APP_ROOT)
                if result.returncode != 0 or not os.path.exists(path):
                    print(f'[TRT] Engine build failed for {os.path.basename(onnx_path)} (exit {result.returncode}), using onnxruntime instead', flush=True)
                    return None
        return TRTSession(path, pin_policy)
    except Exception as e:
        print(f'[TRT] Engine unavailable for {os.path.basename(onnx_path)}, using onnxruntime instead: {e}', flush=True)
        return None


if __name__ == '__main__':
    # Child process entry: python -m modules.trt_engine <model.onnx> <pin_policy>
    import torch  # noqa: F401  (loads the CUDA runtime DLLs)
    _tensorrt_available()
    model_path, policy = sys.argv[1], sys.argv[2]
    target = engine_path(model_path, policy)
    with open(target + '.tmp', 'wb') as f:
        f.write(build_engine(model_path, policy))
    os.replace(target + '.tmp', target)
