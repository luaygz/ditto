"""Download, verify and extract the SWDE dataset into ``data/swde/``; write the manifest.

SWDE: Hao, Cai, Pang, Zhang. "From One Tree to a Forest: a Unified Solution for Structured Web
Data Extraction", SIGIR 2011. The original Microsoft Research / CodePlex host is gone (the
CodePlex archive blob no longer resolves), so the archives come from a GitHub mirror of the
CodePlex project export, pinned to one commit. Each archive is checked three ways: byte size and
git blob SHA-1 against the mirror's tree at that commit, and SHA-256 against the value pinned
here. After extraction the tree is verified by structure (8 verticals x 10 sites, page files
matching the counts in the directory names, one ground-truth file per (vertical, site,
attribute) covering exactly the site's pages). Finally every page's source URL (its leading
``<base href>``) is indexed into ``data/swde/urls/<vertical>-<site>.tsv`` so the evaluation
adapter never rereads 124k pages, and ``split.json`` and ``manifest.json`` are written next to
this script.

Usage::

    uv run python eval/datasets/swde/download.py [--data-dir data/swde] [--force-extract]

Downloads resume from ``*.part`` files. Extraction needs the ``7z`` binary (p7zip). Re-running
is cheap: verified archives are not fetched again and extraction is skipped when the recorded
archive hashes match.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ditto.evaluation.datasets import swde

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "swde"
MANIFEST_PATH = HERE / "manifest.json"
SPLIT_PATH = HERE / "split.json"
FIELDS_DIR = HERE / "fields"
EXTRACTED_MARKER = ".extracted.json"

MIRROR_REPO = "woailaosang/swde"
MIRROR_COMMIT = "e9b60dbbcb899d70e813e8b899b84e3540d540da"
MIRROR_PREFIX = "swde/sourceCode/sourceCode"
BASE_URL = f"https://raw.githubusercontent.com/{MIRROR_REPO}/{MIRROR_COMMIT}/{MIRROR_PREFIX}/"
ORIGINAL_URLS = (
    "http://swde.codeplex.com/",
    "https://archive.codeplex.com/?p=swde",
    "https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip",
)
ALTERNATE_COPY = (
    "https://academictorrents.com/details/411576c7e80787e4b40452360f5f24acba9b5159 "
    "(SWDE_Dataset.zip, 207.31 MB; not used)"
)
PAPER_URL = (
    "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/"
    "StructedDataExtraction_SIGIR2011.pdf"
)
CITATION = (
    "Qiang Hao, Rui Cai, Yanwei Pang, and Lei Zhang. From One Tree to a Forest: a Unified "
    "Solution for Structured Web Data Extraction. SIGIR 2011, pp. 775-784."
)
USER_AGENT = "ditto-eval/1.0 (SWDE dataset download; https://github.com/luaygz/ditto)"
CHUNK = 1 << 20

NOTES = (
    "Original host (CodePlex) unreachable; archives fetched from the GitHub mirror of the CodePlex "
    "project export at the pinned commit and verified against its git tree.",
    "Pages are UTF-8 with a BOM; many still declare <meta charset=iso-8859-1|windows-1252> from "
    "the original crawl. The BOM is authoritative.",
    'Every page starts with <base href="<source URL>"/>; source URLs are unique within each '
    "site and are used as page URLs in the generated urls.tsv (indexed under data/swde/urls/).",
    "Ground-truth files are UTF-8 with a BOM and CRLF line ends; values keep HTML entities as in "
    "the page source (&nbsp;, &amp;, &#150;, ...); the adapter decodes them with html.unescape.",
    "No byte-identical duplicate pages and no missing pages: every site has pages 0000..n-1 and "
    "every ground-truth file covers exactly those ids.",
)


@dataclass(frozen=True)
class Artifact:
    """One file of the mirror: name under ``MIRROR_PREFIX``, size and hashes."""

    name: str
    size: int
    git_blob_sha1: str
    sha256: str
    extract: bool

    @property
    def url(self) -> str:
        return BASE_URL + self.name


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        "auto.7z",
        15380590,
        "daf9684f4e8b11af54c90891004b3f554e73e13a",
        "43494053906dde1ac44c72d30ecdc66aea060b2c02daed3379c88e382f2e79f5",
        True,
    ),
    Artifact(
        "book.7z",
        58095795,
        "9be810d143296c7df1575724f92c752542d52acb",
        "34899c1f7f44b6858c8d1a24924b4b16cf9d9dd85a41a0c30de0315c04569c1f",
        True,
    ),
    Artifact(
        "camera.7z",
        8043333,
        "6af15b5c3878d0ca503d3fe2cdda70136d995d86",
        "549f29713295cff31d0ccaa1420d80efc9c6470affea983770a8db47f149f913",
        True,
    ),
    Artifact(
        "job.7z",
        17631529,
        "f4ce64bf1ae580e951c711ccf482850aeb7140b5",
        "0a0c8e4fabbc7150cf7313acba2386721d9e3011655e0cac79d6140b8acee408",
        True,
    ),
    Artifact(
        "movie.7z",
        34838817,
        "70ab72a557fa99050058c6bafaf041c3c9f3d34e",
        "1323f66fd1d7fcc13d03137fdb6ccbd9faf1e05e882b880981b203fd90d4153f",
        True,
    ),
    Artifact(
        "nbaplayer.7z",
        5674557,
        "8dc54ce01066c6a6e52a5dca45f354c0f2ab2302",
        "34f3e391b6f72cc21c139c5354b0cfd0722830dc3302ba1808f354a9ec2d75f5",
        True,
    ),
    Artifact(
        "restaurant.7z",
        50020553,
        "beb9b4d155ee3f0ab24cd9db41fc411f424aaf21",
        "b4f24a058e11d835b8fff383bb2fa7f68997d0c36ca6086fdf969798ab8b2b4f",
        True,
    ),
    Artifact(
        "university.7z",
        14793527,
        "5edbbeff2f8bb17a2b53670b2620dbf622808ee5",
        "acc7b3df25fc06accadbf1a04108f8edaa8eaecaf0d26b2da8b3276d0fe91ad2",
        True,
    ),
    Artifact(
        "groundtruth.7z",
        2766024,
        "e70630251b62e1836728f09a59d089da8378838e",
        "473d248296878daf2b08f65f0e530f54eb336064d6d4617acf723a12b9d74db9",
        True,
    ),
    Artifact(
        "readme.txt",
        3352,
        "993aa4b44747c16214c33a8007f130ba9575f773",
        "cc50fd75dfc9cbbdd9e8c3dc4d5a33a27d6769f87ce65a53378e689dd822a8c6",
        False,
    ),
    Artifact(
        "website_list.txt",
        4165,
        "e518a97a34a4a3844a12ee6fdf18a2a2ae6e7b91",
        "673dc83abefc558eb0d40aea91a1f05fc59dc11d700b76afc2fe194528b7cfea",
        False,
    ),
)


class VerificationError(RuntimeError):
    """A downloaded file or the extracted tree does not match what is expected."""


def file_hashes(path: Path) -> tuple[str, str]:
    """Return ``(git_blob_sha1, sha256)`` of a file, read once."""
    size = path.stat().st_size
    git = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            git.update(chunk)
            sha.update(chunk)
    return git.hexdigest(), sha.hexdigest()


def verify_file(path: Path, art: Artifact) -> str:
    """Check size, git blob SHA-1 and SHA-256; return the SHA-256."""
    size = path.stat().st_size
    if size != art.size:
        raise VerificationError(f"{art.name}: size {size} != expected {art.size}")
    git_sha, sha = file_hashes(path)
    if git_sha != art.git_blob_sha1:
        raise VerificationError(f"{art.name}: git blob sha1 {git_sha} != {art.git_blob_sha1}")
    if sha != art.sha256:
        raise VerificationError(f"{art.name}: sha256 {sha} != pinned {art.sha256}")
    return sha


def download(
    client: httpx.Client,
    art: Artifact,
    dest: Path,
    *,
    log: Callable[[str], None] = print,
) -> None:
    """Download ``art`` to ``dest``, resuming from ``dest.part`` with an HTTP Range request."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > art.size:
        part.unlink()
        have = 0
    if have < art.size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with client.stream("GET", art.url, headers=headers) as resp:
            if resp.status_code == 200 and have:
                log(f"  {art.name}: server ignored Range, restarting")
                have = 0
            elif resp.status_code not in (200, 206):
                raise VerificationError(f"{art.name}: HTTP {resp.status_code} from {art.url}")
            with part.open("ab" if have else "wb") as fh:
                for chunk in resp.iter_bytes(CHUNK):
                    fh.write(chunk)
    part.replace(dest)


def fetch_all(
    archive_dir: Path,
    *,
    client: httpx.Client,
    artifacts: tuple[Artifact, ...] = ARTIFACTS,
    log: Callable[[str], None] = print,
) -> dict[str, str]:
    """Make every artifact present and verified under ``archive_dir``; return name -> SHA-256."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for art in artifacts:
        dest = archive_dir / art.name
        if dest.exists():
            try:
                hashes[art.name] = verify_file(dest, art)
                log(f"  {art.name}: present, verified")
                continue
            except VerificationError as exc:
                log(f"  {art.name}: {exc}; downloading again")
                dest.unlink()
        log(f"  {art.name}: downloading {art.size / 1e6:.1f} MB")
        download(client, art, dest, log=log)
        try:
            hashes[art.name] = verify_file(dest, art)
        except VerificationError:
            dest.unlink()
            raise
        log(f"  {art.name}: verified")
    return hashes


def seven_zip() -> str:
    for name in ("7z", "7za", "7zz"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("7z not found: install p7zip (e.g. `dnf install p7zip-plugins`)")


def extract_all(
    archive_dir: Path,
    data_dir: Path,
    hashes: dict[str, str],
    *,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> bool:
    """Extract every archive into ``data_dir`` unless the marker records the same hashes;
    return whether anything was extracted."""
    marker = data_dir / EXTRACTED_MARKER
    wanted = {a.name: hashes[a.name] for a in ARTIFACTS if a.extract}
    if not force and marker.is_file() and json.loads(marker.read_text()) == wanted:
        log("  extraction up to date")
        return False
    exe = seven_zip()
    for name in wanted:
        log(f"  extracting {name}")
        subprocess.run(
            [exe, "x", "-y", "-bso0", "-bsp0", f"-o{data_dir}", str(archive_dir / name)],
            check=True,
        )
    marker.write_text(json.dumps(wanted, indent=2) + "\n")
    return True


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative(path: Path) -> str:
    """``path`` relative to the repository root when inside it (manifest paths), else absolute."""
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_manifest(
    scans: list[swde.SiteScan],
    hashes: dict[str, str],
    *,
    downloaded_on: str,
) -> dict[str, Any]:
    split_pages = {"dev": 0, "test": 0}
    for s in scans:
        split_pages[swde.split_of(swde.SiteRef(s.vertical, s.site))] += s.pages
    verticals: dict[str, Any] = {}
    for vertical in swde.VERTICALS:
        rows = [s for s in scans if s.vertical == vertical]
        verticals[vertical] = {
            "sites": len(rows),
            "pages": sum(s.pages for s in rows),
            "attributes": list(swde.ATTRIBUTES[vertical]),
            "site_detail": {
                s.site: {
                    "split": swde.split_of(swde.SiteRef(vertical, s.site)),
                    "directory": f"{vertical}/{s.directory}",
                    "pages": s.pages,
                    "attributes": s.attributes,
                }
                for s in rows
            },
        }
    fields = {
        v: {
            "path": repo_relative(swde.fields_path(v, FIELDS_DIR)),
            "sha256": file_sha256(swde.fields_path(v, FIELDS_DIR)),
        }
        for v in swde.VERTICALS
        if swde.fields_path(v, FIELDS_DIR).is_file()
    }
    return {
        "dataset": "swde",
        "title": "Structured Web Data Extraction (SWDE)",
        "citation": CITATION,
        "paper_url": PAPER_URL,
        "original_urls": list(ORIGINAL_URLS),
        "original_status": "unreachable (CodePlex shut down; archive blob host does not resolve)",
        "alternate_copy": ALTERNATE_COPY,
        "mirror": {
            "repo": f"https://github.com/{MIRROR_REPO}",
            "commit": MIRROR_COMMIT,
            "base_url": BASE_URL,
        },
        "downloaded_on": downloaded_on,
        "archives": [
            {
                "name": a.name,
                "url": a.url,
                "bytes": a.size,
                "sha256": hashes[a.name],
                "git_blob_sha1": a.git_blob_sha1,
            }
            for a in ARTIFACTS
        ],
        "data_dir": "data/swde",
        "layout": {
            "pages": "data/swde/<vertical>/<vertical>-<site>(<n_pages>)/<NNNN>.htm",
            "groundtruth": "data/swde/groundtruth/<vertical>/<vertical>-<site>-<attribute>.txt",
            "page_key": "page file stem, e.g. 0001 (= ground-truth page id)",
        },
        "split": {
            "path": repo_relative(SPLIT_PATH),
            "sha256": file_sha256(SPLIT_PATH),
            "dev_sites": len(swde.sites("dev")),
            "test_sites": len(swde.sites("test")),
            "dev_pages": split_pages["dev"],
            "test_pages": split_pages["test"],
        },
        "fields": fields,
        "totals": {
            "verticals": len(verticals),
            "sites": len(scans),
            "pages": sum(s.pages for s in scans),
            "groundtruth_files": sum(len(s.attributes) for s in scans),
        },
        "verticals": verticals,
        "notes": list(NOTES),
    }


def previous_download_date(hashes: dict[str, str]) -> str | None:
    """Keep the original download date when the manifest already records these archives."""
    if not MANIFEST_PATH.is_file():
        return None
    old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded = {a["name"]: a["sha256"] for a in old.get("archives", [])}
    return str(old["downloaded_on"]) if recorded == hashes else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args(argv)
    data_dir: Path = args.data_dir.resolve()
    archive_dir = data_dir / "_archives"

    print(f"SWDE -> {data_dir}")
    print("[1/5] download + verify")
    with httpx.Client(
        follow_redirects=True, timeout=httpx.Timeout(60.0), headers={"User-Agent": USER_AGENT}
    ) as client:
        hashes = fetch_all(archive_dir, client=client)
    print("[2/5] extract")
    extracted = extract_all(archive_dir, data_dir, hashes, force=args.force_extract)
    print("[3/5] verify layout")
    try:
        scans = swde.scan_layout(data_dir)
    except swde.LayoutError as exc:
        for problem in exc.problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"  {len(scans)} sites, {sum(s.pages for s in scans)} pages: ok")
    print("[4/5] index source URLs")
    refs = swde.sites("all")
    if extracted or not all(swde.url_index_path(data_dir, r).is_file() for r in refs):
        print(f"  {swde.build_url_index(data_dir, refs)} pages indexed")
    else:
        print("  URL index up to date")
    print("[5/5] split.json + manifest.json")
    write_json(SPLIT_PATH, swde.split_document())
    today = dt.datetime.now(dt.UTC).date().isoformat()
    manifest = build_manifest(scans, hashes, downloaded_on=previous_download_date(hashes) or today)
    write_json(MANIFEST_PATH, manifest)
    print(f"  wrote {repo_relative(SPLIT_PATH)}, {repo_relative(MANIFEST_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
