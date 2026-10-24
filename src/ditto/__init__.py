"""ditto: extract a whole downloaded website into JSON with ~one LLM call per page template."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ditto.specgen.models import Spec

try:
    __version__ = version("ditto-extract")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["Spec", "__version__"]
