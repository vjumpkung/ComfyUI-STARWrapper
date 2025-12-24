#!/usr/bin/env python3
"""
Installation script for STARVSRWrapper.
Checks for uv/pip, matches xformers with PyTorch version, and installs requirements.
"""

import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, capture_output=True):
    """Run a command and return the result."""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture_output, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result


def check_uv_available():
    """Check if uv is available in the system."""
    try:
        result = run_command("uv --version", check=False)
        return result.returncode == 0
    except Exception:
        return False


def get_pytorch_version():
    """Get the installed PyTorch version."""
    try:
        import torch

        return torch.__version__
    except ImportError:
        return None


def check_xformers():
    try:
        import xformers

        return True
    except ImportError:
        return False


def get_pytorch_cuda_version():
    """Extract CUDA version from PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            # Get CUDA version from torch (e.g., "cu121" from version string)
            version_str = torch.__version__
            cuda_match = re.search(r"\+cu(\d+)", version_str)
            if cuda_match:
                return f"cu{cuda_match.group(1)}"
            # Fallback: get from torch.version.cuda
            if hasattr(torch.version, "cuda"):
                cuda_ver = torch.version.cuda.replace(".", "")[:3]
                return f"cu{cuda_ver}"
        return None
    except ImportError:
        return None


def find_compatible_xformers(pytorch_version):
    """
    Find compatible xformers version for the given PyTorch version.
    Returns the xformers version string or None if no match found.
    """
    if not pytorch_version:
        print("PyTorch is not installed. Please install PyTorch first.")
        return None

    # Extract version from PyTorch (e.g., "2.1.0+cu121" -> "2.1.0")
    torch_ver_match = re.match(r"(\d+\.\d+\.\d+)", pytorch_version)
    if not torch_ver_match:
        print(f"Could not parse PyTorch version: {pytorch_version}")
        return None

    torch_version = torch_ver_match.group(1)

    # Common xformers compatibility mapping
    # Based on https://github.com/facebookresearch/xformers#installing-xformers
    compatibility_map = {
        "2.6.0": "0.0.29.post2",
        "2.7.0": "0.0.30",
        "2.7.1": "0.0.31.post1",
        "2.8.0": "0.0.32.post2",
        "2.9.0": "0.0.33",
        "2.9.1": "0.0.33.post2",
    }

    xformers_version = compatibility_map.get(torch_version)

    if xformers_version:
        print(
            f"Found compatible xformers version {xformers_version} for PyTorch {pytorch_version}"
        )
        return xformers_version
    else:
        print(f"No xformers mapping found for PyTorch {torch_version}")
        return None


def install_package(package_manager, package_spec):
    """Install a package using the specified package manager."""
    python_exe = sys.executable

    if package_manager == "uv":
        cmd = f'uv pip install "{package_spec}"'
    else:
        cmd = f'"{python_exe}" -m pip install "{package_spec}"'

    print(f"Installing {package_spec}...")
    result = run_command(cmd, check=False, capture_output=False)
    return result.returncode == 0


def main():
    """Main installation routine."""
    print("=" * 60)
    print("STARVSRWrapper Installation Script")
    print("=" * 60)

    # Step 1: Check for uv or fall back to pip
    print("\n[1/3] Checking package manager...")
    use_uv = check_uv_available()
    package_manager = "uv" if use_uv else "pip"
    print(f"Using package manager: {package_manager}")

    # Step 2: Check xformers compatibility
    print("\n[2/3] Checking xformers compatibility...")

    if check_xformers():
        print("xformers exists skip installation")
        return 0

    pytorch_version = get_pytorch_version()
    cuda_version = get_pytorch_cuda_version()

    if pytorch_version:
        print(f"PyTorch version: {pytorch_version}")
        if cuda_version:
            print(f"CUDA version: {cuda_version}")

        xformers_version = find_compatible_xformers(pytorch_version)

        if xformers_version:
            # Try to install xformers
            success = install_package(package_manager, f"xformers=={xformers_version}")
            if not success:
                print("\n⚠️  Xformers installation failed.")
                print("Please install xformers manually:")
                print(f"  {sys.executable} -m pip install xformers=={xformers_version}")
                print(
                    "Or visit: https://github.com/facebookresearch/xformers#installing-xformers"
                )
        else:
            print("\n⚠️  Could not determine compatible xformers version.")
            print("Please install xformers manually:")
            print(
                "  Visit: https://github.com/facebookresearch/xformers#installing-xformers"
            )
    else:
        print("PyTorch not found. Skipping xformers installation.")
        print("Please install PyTorch first, then install xformers manually.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
