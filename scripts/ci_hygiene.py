"""Validate changed-file hygiene without installing repository hooks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import tomli as tomllib
import yaml

TEXT_SUFFIXES = {
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yml",
    ".yaml",
    ".astro",
    ".svg",
}
PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
CONFLICT_MARKER = re.compile(r"^(?:<{7}(?:\s|$)|={7}\s*$|>{7}\s)")


def inspect_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return errors
    if path.stat().st_size > 1024 * 1024:
        errors.append("exceeds 1 MiB changed-file limit")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return errors
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return errors
    if text and not text.endswith("\n"):
        errors.append("missing final newline")
    for number, line in enumerate(text.splitlines(), 1):
        if line.endswith((" ", "\t")):
            errors.append(f"line {number}: trailing whitespace")
        if CONFLICT_MARKER.match(line):
            errors.append(f"line {number}: possible merge-conflict marker")
    if PRIVATE_KEY.search(text):
        errors.append("contains a private-key marker")
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            json.loads(text)
        elif suffix == ".toml":
            tomllib.loads(text)
        elif suffix in {".yml", ".yaml"}:
            list(yaml.safe_load_all(text))
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, yaml.YAMLError) as error:
        errors.append(f"invalid {suffix[1:].upper()} syntax: {error}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    violations = [(path, error) for path in args.paths for error in inspect_file(path)]
    if violations:
        for path, error in violations:
            print(f"{path}: {error}")
        return 1
    print(f"Checked changed-file hygiene for {len(args.paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
