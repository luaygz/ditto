"""``clusters.json``: templates, their signature groups and member pages."""

from __future__ import annotations

import random
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, Field


class SignatureGroup(BaseModel):
    id: int
    sig_hash: str = Field(description="Hex of the 64-bit hash of the page signature.")
    size: int
    template_id: str
    page_ids: list[int]


class Cluster(BaseModel):
    template_id: str = Field(description="t_ + 4+ hex chars, stable for the same input.")
    size: int
    is_tail: bool = Field(description="size < min_cluster_size")
    signature_groups: list[int]
    url_patterns: list[str] = Field(default_factory=list, description="Most common first.")
    url_examples: list[str] = Field(default_factory=list, max_length=10)
    page_ids: list[int]
    merged_from: list[str] = Field(
        default_factory=list, description="Set by the stage-7 merge rule."
    )
    split_from: str | None = Field(None, description="Set by the stage-7 split rule.")


class Clusters(BaseModel):
    version: int = 1
    n_pages: int
    min_cluster_size: int
    params: dict[str, float | int | str] = Field(default_factory=dict)
    clusters: list[Cluster]
    signature_groups: list[SignatureGroup]

    @classmethod
    def load(cls, path: str | Path) -> Clusters:
        return cls.model_validate_json(Path(path).read_bytes())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=1) + "\n")

    @cached_property
    def _by_id(self) -> dict[str, Cluster]:
        return {c.template_id: c for c in self.clusters}

    @cached_property
    def _page_template(self) -> dict[int, str]:
        return {pid: c.template_id for c in self.clusters for pid in c.page_ids}

    def get(self, template_id: str) -> Cluster:
        return self._by_id[template_id]

    def template_of(self, page_id: int) -> str | None:
        return self._page_template.get(page_id)

    def templates(self, *, include_tail: bool = False) -> list[Cluster]:
        """Clusters sorted by size (largest first); tail clusters only when asked."""
        out = [c for c in self.clusters if include_tail or not c.is_tail]
        return sorted(out, key=lambda c: (-c.size, c.template_id))

    def sample(self, template_id: str, n: int = 2000, seed: int = 0) -> list[int]:
        """Up to ``n`` page ids of a cluster, stratified by signature group (proportional, at
        least one page per group while room remains). Deterministic for a given seed."""
        cluster = self.get(template_id)
        if cluster.size <= n:
            return sorted(cluster.page_ids)
        rng = random.Random(f"{seed}:{template_id}")
        groups = [g for g in self.signature_groups if g.template_id == template_id]
        groups.sort(key=lambda g: -g.size)
        picked: list[int] = []
        for g in groups:
            quota = max(1, round(n * g.size / cluster.size))
            take = min(quota, g.size, n - len(picked))
            if take <= 0:
                break
            picked.extend(rng.sample(g.page_ids, take))
        return sorted(picked)
