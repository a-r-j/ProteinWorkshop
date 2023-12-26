import subprocess

import torch
from loguru import logger


def _install_pyg(force_reinstall: bool = False):
    torch_version = torch.__version__
    cuda_version = (
        torch.version.cuda.replace(".", "")
        if torch.cuda.is_available()
        else None
    )
    logger.info(f"Detected PyTorch version: {torch_version}")
    logger.info(f"Detected CUDA version: {cuda_version}")
    logger.info(
        f"Installing PyTorch Geometric for PyTorch {torch_version} and CUDA {cuda_version}"
    )

    if cuda_version is None:
        cuda_version = "cpu"
    else:
        cuda_version = f"cu{cuda_version}"

    if torch_version.startswith("2"):
        # torch_version = "2.0.0"
        if cuda_version == "cu116":
            raise ValueError("PyTorch 2.0.0 does not support CUDA 11.6")
        if torch_version.startswith("2.1"):
            torch_version = "2.1.0"

    if torch_version.startswith("1.13"):
        if cuda_version == "cu118":
            raise ValueError("PyTorch 1.13.0 does not support CUDA 11.8")

    if "+" not in torch_version:
        # Normally torch cpu versions don't have a "+" in them, so we need to
        #  add one to make the url work. The gpu versions typically already
        #  have a "+", so we only add one explicitly if it's missing.
        torch_version += (
            "+cpu" if cuda_version == "cpu" else f"+{cuda_version}"
        )

    # c.f. https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
    #  for detailed install instructions
    command = (
        "pip install --force-reinstall " if force_reinstall else "pip install "
    )
    command += "torch_scatter torch_sparse torch_cluster torch_spline_conv "
    command += f"-f https://data.pyg.org/whl/torch-{torch_version}.html"
    logger.info(f"Running install: {command}")
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    _install_pyg()
