"""Implement CUDA memory monitoring and management utilities."""
import gc

import torch

from beartype.typing import Union
from loguru import logger as log


def gpu_memory_usage(device: Union[int, torch.device] = 0) -> float:
    """
    Get GPU memory usage in GB.
    From: https://github.com/pytorch/pytorch/issues/82218#issuecomment-1675254117

    :param device: GPU device as an index or a `device` object.
    :return: GPU memory usage in GB.
    """
    return torch.cuda.memory_allocated(device) / 1024.0**3


def gpu_memory_usage_all(device: Union[int, torch.device] = 0) -> tuple[float, float]:
    """
    Get GPU memory usage in GB.
    From: https://github.com/pytorch/pytorch/issues/82218#issuecomment-1675254117

    :param device: GPU device as an index or a `device` object.
    :return: GPU memory usage and cache in GB.
    """
    usage = torch.cuda.memory_allocated(device) / 1024.0**3
    reserved = torch.cuda.memory_reserved(device) / 1024.0**3
    cache = reserved - usage
    return usage, cache


def clean_up_torch_gpu_memory(device: Union[int, torch.device] = 0):
    """
    Clean up PyTorch GPU memory systematically.
    From: https://github.com/pytorch/pytorch/issues/82218#issuecomment-1675254117

    :param device: GPU device as an index or a `device` object.
    """
    try:
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        gc.collect()
        torch.cuda.empty_cache()

        if (mem := gpu_memory_usage()) > 3.0:
            log.warning(f"GPU memory usage is still high, with `mem={mem}`!")
            cnt = 0
            for obj in get_tensors():
                obj.detach()
                obj.grad = None
                obj.storage().resize_(0)
                cnt += 1
            gc.collect()
            torch.cuda.empty_cache()
            usage, cache = gpu_memory_usage_all(device=device)
            log.warning(
                f"Forcibly cleared {cnt} tensors: {mem:.03f}GB -> {usage:.03f}GB (+{cache:.03f}GB cache)"
            )


def get_tensors(gpu_only: bool = True):
    """
    Get all tensors in memory.
    From: https://github.com/pytorch/pytorch/issues/82218#issuecomment-1675254117

    :param gpu_only: If True, only return tensors on GPU.
    :return: Generator of tensors.
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda or not gpu_only:
                yield tensor
        except Exception:  # nosec B112 pylint: disable=broad-exception-caught
            continue
