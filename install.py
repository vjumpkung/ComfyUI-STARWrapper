#!/usr/bin/env python3
"""
Installation script for STARVSRWrapper.

Resolves an xformers build that matches the PyTorch already installed in this
environment and installs it with uv (if available) or pip.

Resolution order:
  1. Ask the official PyTorch wheel index for the detected CUDA tag
     (https://download.pytorch.org/whl/cuXXX/xformers/) and pick the newest
     release whose declared `torch` pin is satisfied by the installed PyTorch.
  2. Fall back to a static torch -> xformers table if the index is unreachable.

PyTorch is never installed, upgraded or replaced: the install runs with
--no-deps so a mismatched xformers can't drag a different torch into a working
ComfyUI environment. Failures are reported and the script still exits 0 so it
can never block ComfyUI startup.
"""

import json
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request

PYTORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
PYPI_RELEASE = "https://pypi.org/pypi/xformers/{version}/json"
NETWORK_TIMEOUT = 20

# Fallback table, newest first. Used only when the PyTorch index is unreachable.
# Each entry is the newest xformers release that pins that exact torch version.
FALLBACK_MAP = {
    "2.10.0": "0.0.34",
    "2.9.1": "0.0.33.post2",
    "2.9.0": "0.0.33.post1",
    "2.8.0": "0.0.32.post2",
    "2.7.1": "0.0.31.post1",
    "2.7.0": "0.0.30",
    "2.6.0": "0.0.29.post2",
}
# xformers releases that declare a floor (torch>=X) rather than an exact pin.
FALLBACK_FLOOR = [("2.10", "0.0.35")]


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
    """Get the installed PyTorch version, e.g. '2.10.0+cu130'."""
    try:
        import torch

        return torch.__version__
    except ImportError:
        return None


def base_version(version_string):
    """'2.10.0+cu130' -> '2.10.0'. Returns None if unparseable."""
    if not version_string:
        return None
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_string)
    return match.group(0) if match else None


def version_key(version_string):
    """Sortable key for a release like '0.0.33.post2' or '2.10'."""
    numbers = [int(n) for n in re.findall(r"\d+", version_string)]
    # A plain release outranks nothing, but ".postN" outranks the bare release.
    post = 1 if ".post" in version_string else 0
    return (numbers[:3] + [0, 0, 0])[:3] + [post] + numbers[3:]


def get_pytorch_cuda_tag():
    """Extract the wheel CUDA tag from PyTorch, e.g. 'cu130'. None if CPU/ROCm."""
    try:
        import torch
    except ImportError:
        return None

    local = re.search(r"\+(cu\d+)", torch.__version__)
    if local:
        return local.group(1)
    cuda = getattr(torch.version, "cuda", None)
    if cuda:
        major, _, minor = cuda.partition(".")
        return f"cu{major}{minor or '0'}"
    return None


def wheel_tags_match(filename):
    """Is this wheel installable on the running interpreter and platform?"""
    parts = filename[: -len(".whl")].split("-")
    if len(parts) < 5:
        return False
    python_tag, abi_tag, platform_tag = parts[-3], parts[-2], parts[-1]

    if sys.platform == "win32":
        want_platform = "win_amd64"
    elif sys.platform == "darwin":
        want_platform = "macosx"
    else:
        want_platform = "manylinux"
    if want_platform not in platform_tag:
        return False
    if (
        platform.machine().lower() not in ("amd64", "x86_64")
        and want_platform != "macosx"
    ):
        return False

    # abi3 / py3 wheels work on any newer CPython; cpXY wheels need an exact match.
    exact = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if abi_tag in ("abi3", "none"):
        floor = re.match(r"(?:cp|py)(\d)(\d+)", python_tag)
        if not floor:
            return python_tag in (exact, "py3")
        return (sys.version_info.major, sys.version_info.minor) >= (
            int(floor.group(1)),
            int(floor.group(2)),
        )
    return python_tag == exact


def fetch_index_versions(cuda_tag):
    """Versions on the PyTorch index for this CUDA tag with an installable wheel."""
    url = PYTORCH_INDEX.format(tag=cuda_tag) + "/xformers/"
    try:
        with urllib.request.urlopen(url, timeout=NETWORK_TIMEOUT) as response:
            html = response.read().decode("utf-8", "ignore")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  Could not read {url}: {exc}")
        return []

    versions = set()
    for filename in re.findall(r"xformers-[^\"<>#]+?\.whl", html):
        filename = filename.split("/")[-1]
        if "%2B" in filename or "+" in filename:
            continue  # nightly / local-version build
        if wheel_tags_match(filename):
            versions.add(filename.split("-")[1])
    return sorted(versions, key=version_key, reverse=True)


def fetch_torch_requirement(xformers_version):
    """The `torch` specifier that this xformers release declares, or None."""
    try:
        url = PYPI_RELEASE.format(version=xformers_version)
        with urllib.request.urlopen(url, timeout=NETWORK_TIMEOUT) as response:
            metadata = json.load(response)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return None
    for requirement in metadata.get("info", {}).get("requires_dist") or []:
        match = re.match(r"\s*torch\s*(==|>=)\s*([\d.]+)", requirement)
        if match:
            return match.group(1), match.group(2).rstrip(".")
    return None


def requirement_satisfied(requirement, torch_version):
    """Does the installed torch satisfy ('==' | '>=', version)?"""
    operator, wanted = requirement
    installed = version_key(torch_version)
    target = version_key(wanted)
    if operator == "==":
        # A pin of '2.10.0' matches 2.10.0 only; a pin of '2.10' matches 2.10.x.
        depth = len(re.findall(r"\d+", wanted))
        return installed[:depth] == target[:depth]
    return installed >= target


def resolve_xformers(torch_version, cuda_tag):
    """Pick (version, index_url) for the installed torch, or (None, None)."""
    torch_base = base_version(torch_version)
    if not torch_base:
        print(f"Could not parse PyTorch version: {torch_version}")
        return None, None

    if cuda_tag:
        index_url = PYTORCH_INDEX.format(tag=cuda_tag)
        candidates = fetch_index_versions(cuda_tag)
        if candidates:
            print(
                f"  {len(candidates)} candidate(s) on the {cuda_tag} index: {', '.join(candidates)}"
            )
            for candidate in candidates:
                requirement = fetch_torch_requirement(candidate)
                if requirement is None:
                    continue
                if requirement_satisfied(requirement, torch_base):
                    print(
                        f"  xformers {candidate} declares torch{requirement[0]}{requirement[1]}"
                        f" -> matches {torch_base}"
                    )
                    return candidate, index_url
            print(
                f"  No {cuda_tag} build declares compatibility with torch {torch_base}."
            )
        # Index unreachable or nothing compatible: try the static table below.
    else:
        index_url = None
        print("  PyTorch has no CUDA build tag (CPU or ROCm); using PyPI.")

    static = FALLBACK_MAP.get(torch_base)
    if not static:
        for floor, version in FALLBACK_FLOOR:
            if version_key(torch_base) >= version_key(floor):
                static = version
                break
    if static:
        print(f"  Falling back to the static table: xformers {static}")
        return static, index_url

    print(f"  No known xformers release for PyTorch {torch_base}.")
    return None, None


def get_installed_xformers():
    """(version, torch_requirement) for the installed xformers, or None."""
    try:
        import importlib.metadata as metadata

        version = metadata.version("xformers")
    except Exception:
        return None
    requirement = None
    try:
        for entry in metadata.requires("xformers") or []:
            match = re.match(r"\s*torch\s*(==|>=)\s*([\d.]+)", entry)
            if match:
                requirement = (match.group(1), match.group(2).rstrip("."))
                break
    except Exception:
        pass
    return version, requirement


def install_package(package_manager, package_spec, index_url=None):
    """Install one package without letting it touch the existing torch."""
    if package_manager == "uv":
        cmd = f'uv pip install --no-deps "{package_spec}"'
    else:
        cmd = f'"{sys.executable}" -m pip install --no-deps "{package_spec}"'
    if index_url:
        cmd += f" --index-url {index_url}"

    print(f"Running: {cmd}")
    result = run_command(cmd, check=False, capture_output=False)
    return result.returncode == 0


def main():
    """Main installation routine."""
    print("=" * 60)
    print("STARVSRWrapper Installation Script")
    print("=" * 60)

    print("\n[1/3] Checking package manager...")
    package_manager = "uv" if check_uv_available() else "pip"
    print(f"Using package manager: {package_manager}")

    print("\n[2/3] Inspecting the environment...")
    torch_version = get_pytorch_version()
    if not torch_version:
        print("PyTorch not found. Install PyTorch first, then re-run this script.")
        return 0

    cuda_tag = get_pytorch_cuda_tag()
    torch_base = base_version(torch_version)
    print(f"PyTorch version: {torch_version}")
    print(f"CUDA wheel tag:  {cuda_tag or 'none (CPU/ROCm build)'}")
    print(f"Python:          {sys.version.split()[0]} on {sys.platform}")

    installed = get_installed_xformers()
    if installed:
        version, requirement = installed
        if (
            requirement
            and torch_base
            and not requirement_satisfied(requirement, torch_base)
        ):
            print(
                f"xformers {version} is installed but was built for "
                f"torch{requirement[0]}{requirement[1]}, not {torch_base}. Reinstalling."
            )
        else:
            print(
                f"xformers {version} is installed and matches this PyTorch. Nothing to do."
            )
            return 0

    print("\n[3/3] Resolving a matching xformers build...")
    xformers_version, index_url = resolve_xformers(torch_version, cuda_tag)

    if not xformers_version:
        print("\n[!] Could not determine a compatible xformers version.")
        print("    Install it manually, e.g.:")
        hint = f" --index-url {PYTORCH_INDEX.format(tag=cuda_tag)}" if cuda_tag else ""
        print(f'      "{sys.executable}" -m pip install --no-deps xformers{hint}')
        print(
            "    Version matrix: https://github.com/facebookresearch/xformers#installing-xformers"
        )
        return 0

    if not install_package(package_manager, f"xformers=={xformers_version}", index_url):
        print("\n[!] xformers installation failed.")
        print("    Retry manually:")
        hint = f" --index-url {index_url}" if index_url else ""
        print(
            f'      "{sys.executable}" -m pip install --no-deps '
            f'"xformers=={xformers_version}"{hint}'
        )
        return 0

    print(f"\nInstalled xformers {xformers_version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
