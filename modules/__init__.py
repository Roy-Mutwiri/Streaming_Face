import sys

# basicsr (used by gfpgan) imports torchvision.transforms.functional_tensor, which newer torchvision
# removed. Alias it to torchvision.transforms.functional so no manual patch of basicsr is needed.
# Importing torch here first also loads its CUDA/cuDNN DLLs before onnxruntime, which reuses them.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    try:
        import torchvision.transforms.functional as _functional
        sys.modules['torchvision.transforms.functional_tensor'] = _functional
    except ImportError:
        pass
