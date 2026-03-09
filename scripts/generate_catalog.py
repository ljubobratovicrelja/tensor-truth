#!/usr/bin/env python3
"""Generate catalog.json from extension_library YAML files.

Scans extension_library/{commands,agents}/*.yaml, reads metadata,
and writes extension_library/catalog.json.

Usage:
    python scripts/generate_catalog.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main():
    repo_root = Path(__file__).resolve().parent.parent
    library_dir = repo_root / "extension_library"

    if not library_dir.is_dir():
        print(f"Error: {library_dir} not found", file=sys.stderr)
        sys.exit(1)

    extensions = []
    for ext_type, subdir in [("command", "commands"), ("agent", "agents")]:
        dir_path = library_dir / subdir
        if not dir_path.is_dir():
            continue
        for path in sorted(dir_path.iterdir()):
            if path.suffix not in (".yaml", ".yml"):
                continue
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: failed to parse {path.name}: {e}", file=sys.stderr)
                continue

            if not isinstance(raw, dict):
                continue

            entry = {
                "name": raw.get("name", path.stem),
                "type": ext_type,
                "filename": path.name,
                "description": raw.get("description", ""),
            }
            requires_mcp = raw.get("requires_mcp")
            if requires_mcp:
                entry["requires_mcp"] = requires_mcp

            extensions.append(entry)

    catalog = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extensions": extensions,
    }

    output_path = library_dir / "catalog.json"
    output_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {output_path} with {len(extensions)} extensions")


if __name__ == "__main__":
    main()
