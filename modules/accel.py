"""
GPU acceleration helpers.

When the CUDA execution provider is selected and TensorRT is installed, models are run
through TensorRT (FP16) with a disk engine cache. TensorRT is 2-5x faster than plain CUDA
on RTX cards. The first launch builds the engines (several minutes), later launches load
them from models/trt_cache in seconds.
"""
import os
from typing import Any, List

import modules.globals
from modules.utilities import resolve_relative_path

TRT_CACHE_DIR = resolve_relative_path('../models/trt_cache')
_trt_available = None


def _tensorrt_available() -> bool:
    global _trt_available
    if _trt_available is None:
        _trt_available = False
        try:
            import tensorrt_libs  # pip package tensorrt-cu12 ships the TensorRT DLLs here
            libs_dir = os.path.dirname(tensorrt_libs.__file__)
            os.environ['PATH'] = libs_dir + os.pathsep + os.environ.get('PATH', '')
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(libs_dir)
            import onnxruntime
            _trt_available = 'TensorrtExecutionProvider' in onnxruntime.get_available_providers()
        except Exception:
            pass
    return _trt_available


def get_providers(fp16: bool = True) -> List[Any]:
    """
    Execution providers for an onnxruntime session, upgraded to TensorRT when possible.
    fp16=False keeps TensorRT in FP32/TF32 for models whose output degrades in FP16 (GPEN).
    """
    providers = list(modules.globals.execution_providers)
    if 'CUDAExecutionProvider' not in providers or not modules.globals.use_tensorrt or not _tensorrt_available():
        return providers
    os.makedirs(TRT_CACHE_DIR, exist_ok=True)
    trt_options = {
        'trt_fp16_enable': fp16,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': TRT_CACHE_DIR,
        'trt_timing_cache_enable': True,
        'trt_timing_cache_path': TRT_CACHE_DIR,
        'trt_max_workspace_size': 4 * 1024 ** 3,
    }
    return [('TensorrtExecutionProvider', trt_options)] + providers
