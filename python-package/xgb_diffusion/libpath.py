# coding: utf-8
"""Find the path to the bundled XGBoost dynamic library."""

import os
import platform
import sys
from typing import List


class XGBoostLibraryNotFound(Exception):
    """Error thrown when the bundled native library is not found."""


def is_sphinx_build() -> bool:
    """`XGBOOST_BUILD_DOC` is used by the sphinx conf.py to skip building the C++ code."""
    return bool(os.environ.get("XGBOOST_BUILD_DOC", False))


def find_lib_path() -> List[str]:
    """Find the path to the bundled XGBoost dynamic library.

    Returns
    -------
    lib_path
       List of all found library paths.
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [
        # normal, after installation `lib` is copied into Python package tree.
        os.path.join(curr_path, "lib"),
        # editable installation, no copying is performed.
        os.path.join(curr_path, os.path.pardir, os.path.pardir, "lib"),
    ]

    if sys.platform == "win32":
        dll_path = [os.path.join(p, "xgboost.dll") for p in dll_path]
    elif sys.platform.startswith(("linux", "freebsd", "emscripten")):
        dll_path = [os.path.join(p, "libxgboost.so") for p in dll_path]
    elif sys.platform == "darwin":
        dll_path = [os.path.join(p, "libxgboost.dylib") for p in dll_path]
    elif sys.platform == "cygwin":
        dll_path = [os.path.join(p, "cygxgboost.dll") for p in dll_path]
    if platform.system() == "OS400":
        dll_path = [os.path.join(p, "libxgboost.so") for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_path and not is_sphinx_build():
        msg = (
            "Cannot find the bundled xgb-diffusion native library in the candidate path.  "
            + "List of candidates:\n- "
            + ("\n- ".join(dll_path))
            + "\nxgb-diffusion Python package path: "
            + curr_path
            + "\nsys.base_prefix: "
            + sys.base_prefix
        )
        raise XGBoostLibraryNotFound(msg)
    return lib_path
