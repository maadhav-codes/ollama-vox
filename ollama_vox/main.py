"""Application entry point and bootstrap logic for Ollama Vox.

This module is the first thing that runs when the user executes
``ollama-vox`` (or ``python -m ollama_vox.main``). It is responsible for:

1. **Parsing CLI arguments** — currently just ``--setup`` for first-run model download.
2. **Setting up logging** — structured log lines with timestamps and levels.
3. **Loading configuration** — reads ``config.yaml`` via
   :class:`~ollama_vox.core.config.AppConfig`.
4. **Running startup health checks** — verifies that model files exist and Ollama is reachable before starting the GUI.
5. **Constructing the pipeline** — wires together
   :class:`~ollama_vox.core.audio.AudioRecorder`,
   :class:`~ollama_vox.core.stt.STT`,
   :class:`~ollama_vox.core.llm.OllamaClient`,
   :class:`~ollama_vox.core.tts.TTS`, and
   :class:`~ollama_vox.core.workers.Pipeline`.
6. **Launching the system-tray UI** — hands control to the Qt event loop.

espeak / phonemizer compatibility patch
-----------------------------------------
The Kokoro TTS model optionally uses the ``misaki.espeak`` G2P (grapheme-to-phoneme) backend, which in turn calls ``phonemizer``. Older versions of ``phonemizer`` may not expose ``EspeakWrapper.set_data_path``. The try/except block at module level monkey-patches the method if it is missing, so the import always succeeds regardless of the installed ``phonemizer`` version.
"""

import argparse
import logging
import os

import requests
import yaml

# --- Optional espeak / phonemizer compatibility patch ---
# Try to import misaki's espeak backend. If EspeakWrapper doesn't have
# set_data_path (older phonemizer), we inject a shim that sets the
# ESPEAK_DATA_PATH environment variable instead.
try:
    import dateutil.parser
    import phonemizer
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if not hasattr(EspeakWrapper, "set_data_path"):
        # The method is missing — create a minimal replacement.
        def _set_data_path(path):
            """Fallback that stores the espeak data path as an env variable."""
            os.environ["ESPEAK_DATA_PATH"] = str(path)

        # Attach the shim to the class so callers use a consistent API.
        EspeakWrapper.set_data_path = _set_data_path

    import misaki.espeak
except ImportError:
    # phonemizer / misaki are not installed. This is fine — TTS will use
    # a different G2P backend (or none at all for supported voices).
    EspeakWrapper = None

from ollama_vox.core.audio import AudioRecorder
from ollama_vox.core.config import AppConfig, ConfigValidationError
from ollama_vox.core.llm import OllamaClient
from ollama_vox.core.stt import STT
from ollama_vox.core.tts import TTS
from ollama_vox.core.workers import Pipeline
from ollama_vox.ui.model_setup import OllamaModelWizard
from ollama_vox.ui.tray_app import VoiceTrayApp as VoiceApp


def load_config() -> AppConfig:
    """Locate and parse ``config.yaml``, returning a validated AppConfig.

    Search order for ``config.yaml``:

    1. Same directory as this file (i.e. ``ollama_vox/config.yaml``), which is where it lives when the package is installed via pip.
    2. The current working directory (``./config.yaml``), useful when running from the source tree.

    Returns:
        AppConfig: A fully validated configuration object.

    Raises:
        SystemExit: If the config file is not found or contains validation errors. Prints a human-friendly error message before exiting.
    """
    from pathlib import Path

    # __file__ is the path to this module (main.py).
    # .parent resolves to the ollama_vox/ package directory.
    base_dir = Path(__file__).parent.resolve()
    config_path = base_dir / "config.yaml"

    # Fall back to the CWD if the package directory doesn't have the file
    # (can happen in some development setups).
    if not config_path.exists():
        config_path = Path("config.yaml")

    try:
        with open(config_path) as f:
            # yaml.safe_load returns a dict (or None for empty files).
            # We coerce None to {} so AppConfig.from_dict always gets a dict.
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Configuration Error: Could not find config.yaml at {config_path}")
        import sys

        sys.exit(1)

    try:
        return AppConfig.from_dict(data)
    except ConfigValidationError as e:
        print(f"Configuration Error: {e}")
        import sys

        sys.exit(1)


def configure_logging() -> None:
    """Configure the root Python logger with a structured log format.

    Sets the log level to INFO (so DEBUG messages are hidden by default) and uses a log format that matches common structured-logging conventions (``key=value`` pairs), making it easy to parse with tools like ``grep`` or a log aggregation system.

    The format emits:
        ``ts=<timestamp> level=<LEVEL> logger=<module> msg=<message>``
    """
    logging.basicConfig(
        level=logging.INFO,
        format="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
    )


def run_startup_health_checks(config: AppConfig) -> None:
    """Verify that required models and services are available before starting.

    Checks three things:

    1. **STT model path** — the Whisper model directory must exist on disk.
    2. **TTS model path** — the Kokoro model directory must exist on disk.
    3. **Python dependencies** — ``mlx_whisper`` and ``mlx_audio`` must be
       importable.
    4. **Ollama server** — sends a GET request to ``/api/tags`` and checks
       for a successful HTTP response.

    This does **not** exit the application on failure. Instead, errors are
    collected and logged at ERROR level. The application continues to start
    (the user can still fix things via the UI or CLI), but the log provides
    a clear record of what went wrong.

    Args:
        config (AppConfig): The validated application configuration, used to
            read model paths and the Ollama endpoint URL.

    Side effects:
        Logs either ``event=startup_health_ok`` or
        ``event=startup_health_failed errors=<list>`` at the appropriate level.
    """
    errors = []

    # --- Check model files ---
    tts_model = config.tts.model
    stt_model = config.stt.model

    if tts_model and not os.path.exists(tts_model):
        errors.append(f"missing_tts_model:{tts_model}")
    if stt_model and not os.path.exists(stt_model):
        errors.append(f"missing_stt_model:{stt_model}")

    # --- Check Python dependencies ---
    try:
        import mlx_audio  # noqa: F401
        import mlx_whisper  # noqa: F401 (imported for side-effect check only)
    except Exception as exc:
        errors.append(f"missing_dependency:{exc}")

    # --- Check Ollama server reachability ---
    ollama_endpoint = config.ollama.endpoint
    try:
        # /api/tags returns the list of locally available models.
        # A 2-second timeout is short enough to fail fast without making
        # startup feel slow.
        r = requests.get(f"{ollama_endpoint}/api/tags", timeout=2)
        r.raise_for_status()
    except Exception as exc:
        errors.append(f"ollama_unreachable:{exc}")

    if errors:
        logging.getLogger(__name__).error(
            "event=startup_health_failed errors=%s", errors
        )
    else:
        logging.getLogger(__name__).info("event=startup_health_ok")


def main() -> None:
    """Bootstrap and launch the Ollama Vox voice assistant.

    This function is the application's main entry point, registered as the
    ``ollama-vox`` console script in ``pyproject.toml``. It performs every
    initialisation step in order:

    1. Parse CLI arguments.
    2. Configure logging.
    3. Load and validate ``config.yaml``.
    4. Create the Qt application instance.
    5. If ``--setup`` was passed, run the model download wizards and exit.
    6. Otherwise, run the Ollama model selection wizard (required).
    7. Run startup health checks (informational — does not block launch).
    8. Construct all pipeline components with settings from config.
    9. Start the pipeline worker threads.
    10. Launch the system-tray UI and enter the Qt event loop.

    The function blocks until the user quits (either from the tray menu or
    with Ctrl-C).

    CLI arguments:
        --setup: Download the recommended Whisper (STT) and Kokoro (TTS) model assets and configure the Ollama model, then exit. Useful for first-time setup or re-downloading models.
    """
    # --- Step 1: parse CLI arguments ---
    parser = argparse.ArgumentParser(description="Ollama Vox")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Download recommended Whisper + Kokoro model assets, then exit.",
    )
    args = parser.parse_args()

    # --- Step 2 & 3: logging + config ---
    configure_logging()
    config = load_config()

    # --- Step 4: Qt application ---
    import sys

    from PySide6.QtWidgets import QApplication

    # QApplication.instance() returns the existing app if one already exists,
    # which avoids the "only one QApplication" restriction during testing.
    _qt_app = QApplication.instance() or QApplication(sys.argv)

    # --- Step 5: --setup mode ---
    from ollama_vox.ui.setup_wizard import AppSetupWizard

    if args.setup:
        # Run model download wizards and exit immediately.
        app_wizard = AppSetupWizard(config)
        app_wizard.run(force_setup=True)
        wizard = OllamaModelWizard(config)
        wizard.run(force_setup=True)
        return

    # --- Step 6: Ollama model selection (required to proceed) ---
    wizard = OllamaModelWizard(config)
    if not wizard.run():
        # User cancelled or Ollama is not running — exit cleanly.
        sys.exit(1)

    # --- Step 7: health checks (informational) ---
    run_startup_health_checks(config)

    # --- Step 8: construct pipeline components ---
    audio_cfg = config.audio
    recorder = AudioRecorder(
        sample_rate=audio_cfg.sample_rate,
        vad_enabled=audio_cfg.vad_enabled,
        vad_threshold=audio_cfg.vad_threshold,
        vad_silence_seconds=audio_cfg.vad_silence_seconds,
        max_duration_seconds=audio_cfg.max_duration_seconds,
    )

    stt = STT(config.stt.model)

    llm = OllamaClient(
        endpoint=config.ollama.endpoint,
        model=config.ollama.model,
        temperature=config.ollama.temperature,
    )

    # Build the style_map for TTS from the config's styles dict.
    # Only include style attributes that are not None (to use TTS defaults for
    # unspecified fields). The dict comprehension filters out None values.
    tts = TTS(
        voice=config.tts.voice,
        rate=config.tts.rate,
        model_id=config.tts.model,
        sample_rate=config.tts.sample_rate,
        split_chars=config.tts.split_chars,
        style_map={
            k: {
                # Filter out None-valued attributes so TTS uses its own defaults.
                sk: sv
                for sk, sv in {
                    "speed": v.speed,
                    "pitch": v.pitch,
                    "voice": v.voice,
                }.items()
                if sv is not None
            }
            for k, v in config.styles.items()
        },
    )

    queue_cfg = config.queue
    pipeline = Pipeline(
        stt,
        llm,
        tts,
        audio_cfg.sample_rate,
        queue_maxsize=queue_cfg.maxsize,
        drop_policy=queue_cfg.drop_policy,
        response_style=config.response_style,
    )

    # --- Step 9: start worker threads ---
    pipeline.start()

    # --- Step 10: launch UI ---
    app = VoiceApp(pipeline, recorder)
    # Wire pipeline callbacks to the tray app's thread-safe signal slots.
    pipeline.set_status_callback(app.set_status)
    pipeline.set_metrics_callback(app.set_metrics)
    # run() shows the tray icon and enters the Qt event loop (blocks here).
    app.run()


# Standard Python idiom: only run main() when this file is executed directly,
# not when it is imported as a module (e.g. in tests).
if __name__ == "__main__":
    main()
