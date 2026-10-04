"""Find existing model assets without loading models or accessing the network."""

import json
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
from platformdirs import (
    user_data_path,
    user_desktop_path,
    user_documents_path,
    user_downloads_path,
)

from ollama_vox.core.config import model_data_dir, save_model_paths


def model_files_available(path: str | Path, kind: str, voice: str = "af_bella") -> bool:
    directory = Path(path)
    try:
        if not isinstance(json.loads((directory / "config.json").read_text()), dict):
            return False
        weights = (
            [directory / "weights.npz"]
            if kind == "stt"
            else [*directory.glob("*.safetensors"), *directory.glob("*.pth")]
        )
        if not any(file.is_file() and file.stat().st_size > 0 for file in weights):
            return False
        return kind == "stt" or any(
            file.is_file() and file.stat().st_size > 0
            for file in (
                directory / "voices" / f"{voice}.safetensors",
                directory / "voices" / f"{voice}.pt",
            )
        )
    except (OSError, ValueError):
        return False


def model_search_roots() -> list[Path]:
    """Known locations only: avoid an unbounded scan of the user's disk."""
    return [
        model_data_dir(),
        Path(__file__).resolve().parents[2],
        Path.cwd(),
        user_documents_path(),
        user_desktop_path(),
        user_downloads_path(),
        user_data_path("ollama-vox", appauthor=False) / "models",
    ]


def discover_model_paths(config, extra_roots: tuple[Path, ...] = ()) -> bool:
    """Reuse complete local models and remember each model's absolute path."""
    changed = False
    for kind, section, repo in (
        ("stt", config.stt, "models--mlx-community--whisper-small.en-mlx-q4"),
        ("tts", config.tts, "models--mlx-community--Kokoro-82M-4bit"),
    ):
        if model_files_available(section.model, kind, config.tts.voice):
            continue
        original = Path(section.original_model).expanduser()
        if original.is_absolute():
            continue  # An explicit config path takes precedence over discovery.
        candidates = []
        for root in [*extra_roots, *model_search_roots()]:
            candidates.extend([root / original, root / original.name, root])
            # Common layouts: Desktop/My Models/... and Documents/ollama-vox/models/...
            for pattern in ("*", "*/models"):
                try:
                    folders = list(root.glob(pattern))
                except OSError:
                    continue
                for folder in folders:
                    if folder.is_dir():
                        candidates.extend(
                            [folder / original, folder / original.name, folder]
                        )
        candidates.extend((Path(HF_HUB_CACHE) / repo / "snapshots").glob("*"))
        for candidate in dict.fromkeys(candidates):
            if model_files_available(candidate, kind, config.tts.voice):
                section.model = str(candidate.resolve())
                changed = True
                break
    if changed:
        save_model_paths(config)
    return changed
