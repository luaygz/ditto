"""Synthetic site generator (``ditto synth generate``): templated pages with ground truth."""

from __future__ import annotations

from ditto.synth.generate import SynthSummary, generate_site, synth_main
from ditto.synth.params import SynthParams, load_params

__all__ = ["SynthParams", "SynthSummary", "generate_site", "load_params", "synth_main"]
