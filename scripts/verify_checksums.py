"""Create or verify the repository-wide SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "CHECKSUMS.sha256"
LINE = re.compile(r"^([0-9a-f]{64})  ([^\\].*)$")


def _git_paths(*arguments: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", *arguments, "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw)


def _tracked_files() -> list[Path]:
    return [path for path in _git_paths("--cached") if path.as_posix() != MANIFEST.name]


def _untracked_files() -> list[Path]:
    return _git_paths("--others", "--exclude-standard")


def _validated_file(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"Unsafe checksum path: {path}")
    candidate = ROOT / path
    try:
        candidate.resolve(strict=True).relative_to(ROOT.resolve(strict=True))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Missing or unsafe checksum path: {path}") from error
    try:
        attributes = getattr(os.lstat(candidate), "st_file_attributes", 0)
    except OSError as error:
        raise SystemExit(f"Cannot inspect checksum path: {path}") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if candidate.is_symlink() or (reparse_flag and attributes & reparse_flag):
        raise SystemExit(f"Refusing to hash symlink or reparse point: {path}")
    if not candidate.is_file():
        raise SystemExit(f"Checksum path is not a regular file: {path}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest() -> None:
    untracked = _untracked_files()
    if untracked:
        names = [path.as_posix() for path in untracked]
        raise SystemExit(f"Refusing to generate checksums with untracked files: {names}")
    lines = [f"{_sha256(_validated_file(path))}  {path.as_posix()}" for path in _tracked_files()]
    temporary = MANIFEST.with_name(f".{MANIFEST.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, MANIFEST)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Wrote {len(lines)} checksums to {MANIFEST.name}")


def verify_manifest() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing {MANIFEST.name}; run with --write to create it")
    expected: dict[Path, str] = {}
    for number, raw in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
        match = LINE.fullmatch(raw)
        if match is None:
            raise SystemExit(f"Invalid checksum record at line {number}")
        path = Path(match.group(2))
        if path.is_absolute() or ".." in path.parts or path in expected:
            raise SystemExit(f"Unsafe or duplicate checksum path at line {number}: {path}")
        expected[path] = match.group(1)

    recorded = set(expected)
    if (ROOT / ".git").exists():
        current = set(_tracked_files())
        if current != recorded:
            missing = sorted(path.as_posix() for path in current - recorded)
            extra = sorted(path.as_posix() for path in recorded - current)
            raise SystemExit(f"Checksum inventory mismatch; missing={missing}, extra={extra}")

    failures = [
        path.as_posix()
        for path, digest in expected.items()
        if _sha256(_validated_file(path)) != digest
    ]
    if failures:
        raise SystemExit(f"Checksum mismatch: {failures}")
    print(f"Verified {len(expected)} repository checksums")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace the checksum manifest")
    args = parser.parse_args()
    if args.write:
        write_manifest()
    else:
        verify_manifest()


if __name__ == "__main__":
    main()
