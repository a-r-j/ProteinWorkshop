import torch
from beartype import beartype
from jaxtyping import jaxtyped


@jaxtyped
@beartype
def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    From https://github.com/drorlab/gvp-pytorch
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )
