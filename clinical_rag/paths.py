import os
import sys


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def setup_paths(project_root: str | None = None) -> str:
    """Add project root and vendor packages to sys.path."""
    root = project_root or get_project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    for vendor_subdir in ("trial2vec", "primekg"):
        vendor_path = os.path.join(root, "vendor", vendor_subdir)
        if os.path.isdir(vendor_path) and vendor_path not in sys.path:
            sys.path.append(vendor_path)

    return root
