import torch
from beartype import beartype as typechecker
from jaxtyping import jaxtyped


@jaxtyped(typechecker=typechecker)
def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Safely normalize a Tensor. Adapted from:
    https://github.com/drorlab/gvp-pytorch.

    :param tensor: Tensor of any shape.
    :type tensor: Tensor
    :param dim: The dimension over which to normalize the input Tensor.
    :type dim: int, optional
    :return: The normalized Tensor.
    :rtype: torch.Tensor
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )
