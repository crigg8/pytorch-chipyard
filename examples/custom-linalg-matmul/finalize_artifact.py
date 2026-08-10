#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


def _has_object_metadata(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    metadata_groups: list[Any] = []
    selected = entry.get("selected_launch")
    if isinstance(selected, dict):
        metadata_groups.append(selected.get("metadata_group"))
    for variant in entry.get("compiled_variants", []):
        if isinstance(variant, dict):
            metadata_groups.append(variant.get("metadata_group"))
    return any(
        isinstance(group, dict)
        and any(str(name).endswith((".o", ".obj")) for name in group)
        for group in metadata_groups
    )


def _set_shell_array(source: str, name: str, entries: list[str]) -> str:
    replacement = [f"{name}=()"]
    replacement.extend(f'{name}+=("{entry}")' for entry in entries)
    pattern = re.compile(
        rf"^{re.escape(name)}=\(\)(?:\n{re.escape(name)}\+=\([^\n]*\))*",
        re.MULTILINE,
    )
    updated, count = pattern.subn("\n".join(replacement), source, count=1)
    if count != 1:
        raise RuntimeError(f"could not find generated {name} array in build.sh")
    return updated


def _load_custom_library(config_path: Path) -> tuple[Path, dict[str, str]]:
    config_bytes = config_path.read_bytes()
    document = json.loads(config_bytes.decode("utf-8"))
    library_value = document.get("library") if isinstance(document, dict) else None
    if not isinstance(library_value, str) or not library_value:
        raise ValueError(f"invalid custom Linalg library in {config_path}")
    library_path = Path(library_value).expanduser()
    if not library_path.is_absolute():
        library_path = config_path.parent / library_path
    library_path = library_path.resolve()
    if not library_path.is_file():
        raise FileNotFoundError(library_path)
    return library_path, {
        "sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def finalize_artifact(
    artifact_dir: Path,
    config_path: Path | None = None,
    library_path: Path | None = None,
) -> bool:
    artifact_dir = artifact_dir.resolve()
    model_spec_path = artifact_dir / "model_spec.json"
    build_script_path = artifact_dir / "build.sh"
    model_spec = json.loads(model_spec_path.read_text(encoding="utf-8"))

    kernels = [
        entry for entry in model_spec.get("kernels", []) if isinstance(entry, dict)
    ]
    missing = [entry for entry in kernels if not _has_object_metadata(entry)]
    libraries = model_spec.get("triton_custom_libraries", [])
    if not isinstance(libraries, list):
        libraries = []
    if not missing and libraries:
        return False
    if len(missing) != 1:
        raise RuntimeError(
            "post-compile fallback requires exactly one kernel without object "
            f"metadata, found {len(missing)}"
        )

    source_object = artifact_dir / "log-ttshared.o"
    if not source_object.is_file():
        raise FileNotFoundError(source_object)
    if library_path is not None:
        library_path = library_path.resolve()
        if not library_path.is_file():
            raise FileNotFoundError(library_path)
        library_metadata = {
            "sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "config_path": "",
            "config_sha256": "",
        }
    elif config_path is not None:
        library_path, library_metadata = _load_custom_library(
            config_path.resolve()
        )
    else:
        raise ValueError("either config_path or library_path is required")

    staged_dir = artifact_dir / "kernels" / "postcompile"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_object = staged_dir / source_object.name
    staged_library = staged_dir / library_path.name
    if source_object != staged_object:
        shutil.copy2(source_object, staged_object)
    if library_path != staged_library:
        shutil.copy2(library_path, staged_library)

    object_group = {staged_object.name: str(staged_object)}
    kernel_entry = missing[0]
    kernel_entry.pop("module_load_error", None)
    kernel_entry["compiled_kernel_source"] = "example_post_async_compile"
    kernel_entry["compiled_variants"] = [{"metadata_group": object_group}]
    kernel_entry["selected_launch"] = {"metadata_group": object_group}
    model_spec["triton_custom_libraries"] = [
        {
            "path": str(staged_library),
            "basename": staged_library.name,
            **library_metadata,
        }
    ]

    model_spec_path.write_text(
        json.dumps(model_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_source = build_script_path.read_text(encoding="utf-8")
    build_source = _set_shell_array(
        build_source,
        "KERNEL_OBJECTS",
        [f"${{SCRIPT_DIR}}/{staged_object.relative_to(artifact_dir)}"],
    )
    build_source = _set_shell_array(
        build_source,
        "TRITON_CUSTOM_LIBRARIES",
        [f"${{SCRIPT_DIR}}/{staged_library.relative_to(artifact_dir)}"],
    )
    build_source, count = re.subn(
        r'\necho "Missing kernel object\(s\) for: [^"\n]*" >&2\nexit 1\n',
        "\n",
        build_source,
        count=1,
    )
    if count != 1:
        raise RuntimeError("could not find generated missing-kernel guard in build.sh")
    build_script_path.write_text(build_source, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize a one-kernel libtriton_chipyard test artifact."
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--library", type=Path)
    args = parser.parse_args()
    changed = finalize_artifact(
        args.artifact_dir,
        config_path=args.config,
        library_path=args.library,
    )
    print("artifact finalized" if changed else "artifact already finalized")


if __name__ == "__main__":
    main()
