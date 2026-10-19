"""
Mixed-precision TensorRT engine for inswapper_128, which breaks in plain FP16.

inswapper_128: its instance-norm layers square activations of up to ~1500, so the variance reaches
millions and overflows FP16 (max 65504). Plain FP16 (inswapper_128_fp16.onnx, or TensorRT's FP16
mode) produces smeared, blocky faces.

Convolutions and matrix multiplies run in FP16 (where the speed is) and the numerically sensitive
layers are pinned to FP32, which gives the FP32 result at close to FP16 speed.
"""
import hashlib
import os
import threading
from typing import Dict, List

import numpy as np

from modules.accel import TRT_CACHE_DIR, _tensorrt_available

FP16_LAYER_TYPES = ('CONVOLUTION', 'DECONVOLUTION', 'MATRIX_MULTIPLY', 'FULLY_CONNECTED')
LOGGER = None  # TensorRT keeps one global logger; creating more just triggers a warning


class TRTSession:
    """Minimal drop-in for the parts of onnxruntime.InferenceSession that insightface's INSwapper uses."""

    def __init__(self, onnx_path: str, pin_policy: str = 'stats'):
        import tensorrt as trt
        import torch
        self.trt, self.torch = trt, torch
        self.lock = threading.Lock()
        global LOGGER
        if LOGGER is None:
            LOGGER = trt.Logger(trt.Logger.WARNING)
        logger = LOGGER
        self.pin_policy = pin_policy
        engine_path = self._engine_path(onnx_path, trt.__version__, torch.cuda.get_device_name(0), pin_policy)
        if not os.path.exists(engine_path):
            print(f'[TRT] Building mixed-precision engine for {os.path.basename(onnx_path)} (one time, a few minutes)...', flush=True)
            with open(engine_path + '.tmp', 'wb') as f:
                f.write(self._build(onnx_path, logger))
            os.replace(engine_path + '.tmp', engine_path)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.buffers: Dict[str, 'torch.Tensor'] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = torch.float32 if self.engine.get_tensor_dtype(name) == trt.float32 else torch.float16
            self.buffers[name] = torch.empty(shape, dtype=dtype, device='cuda')
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    @staticmethod
    def _engine_path(onnx_path: str, trt_version: str, gpu: str, pin_policy: str) -> str:
        stat = os.stat(onnx_path)
        key = f'{os.path.basename(onnx_path)}|{stat.st_size}|{trt_version}|{gpu}|{pin_policy}|mixed-v3'
        os.makedirs(TRT_CACHE_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        return os.path.join(TRT_CACHE_DIR, f'{stem}_mixed_{hashlib.md5(key.encode()).hexdigest()[:16]}.engine')

    @staticmethod
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

    def _pin_to_fp32(self, index: int, layer) -> bool:
        if layer.type.name in FP16_LAYER_TYPES or layer.type.name == 'CONSTANT':
            return False
        float_types = (self.trt.float32, self.trt.float16)
        if not all(layer.get_output(j).dtype in float_types for j in range(layer.num_outputs)):
            return False  # shape/index math stays as it is
        return index in self.stats

    def _build(self, onnx_path: str, logger) -> bytes:
        trt = self.trt
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            raise RuntimeError('TensorRT could not parse ' + onnx_path)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        self.stats = self._stats_layers(network)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if self._pin_to_fp32(i, layer):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError('TensorRT engine build failed')
        return bytes(engine)

    # --- onnxruntime.InferenceSession-like API ---
    def get_inputs(self):
        return [_IO(n, list(self.buffers[n].shape)) for n in self.input_names]

    def get_outputs(self):
        return [_IO(n, list(self.buffers[n].shape)) for n in self.output_names]

    def get_providers(self):
        return ['TensorRT (mixed FP16/FP32)']

    def run(self, output_names, feeds):
        torch = self.torch
        with self.lock, torch.cuda.stream(self.stream):
            for name, value in feeds.items():
                self.buffers[name].copy_(torch.from_numpy(np.ascontiguousarray(value)), non_blocking=False)
            self.context.execute_async_v3(self.stream.cuda_stream)
            results = [self.buffers[name].float().cpu().numpy() for name in (output_names or self.output_names)]
            self.stream.synchronize()
        return results


class _IO:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


def create_session(onnx_path: str, pin_policy: str):
    """Returns a mixed-precision TensorRT session, or None if TensorRT is unavailable or the build fails."""
    if not _tensorrt_available():
        return None
    try:
        return TRTSession(onnx_path, pin_policy)
    except Exception as e:
        print(f'[TRT] Mixed-precision engine unavailable for {os.path.basename(onnx_path)}, falling back to onnxruntime: {e}', flush=True)
        return None
