import argparse
import yaml
import logging
import os
import requests

try:
    import dateutil.parser
    import phonemizer
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if not hasattr(EspeakWrapper, "set_data_path"):

        def _set_data_path(path):
            os.environ["ESPEAK_DATA_PATH"] = str(path)

        EspeakWrapper.set_data_path = _set_data_path
    import misaki.espeak
except ImportError:
    EspeakWrapper = None

from ollama_vox.core.audio import AudioRecorder
from ollama_vox.core.stt import STT
from ollama_vox.core.llm import OllamaClient
from ollama_vox.core.tts import TTS
from ollama_vox.core.workers import Pipeline
from ollama_vox.ui.tray_app import VoiceTrayApp as VoiceApp
from ollama_vox.core.config import AppConfig, ConfigValidationError
from ollama_vox.ui.model_setup import OllamaModelWizard


def load_config():
    from pathlib import Path

    base_dir = Path(__file__).parent.resolve()
    config_path = base_dir / "config.yaml"

    if not config_path.exists():
        config_path = Path("config.yaml")

    try:
        with open(config_path) as f:
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


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
    )


def run_startup_health_checks(config):
    errors = []
    tts_model = config.tts.model
    stt_model = config.stt.model
    if tts_model and not os.path.exists(tts_model):
        errors.append(f"missing_tts_model:{tts_model}")
    if stt_model and not os.path.exists(stt_model):
        errors.append(f"missing_stt_model:{stt_model}")
    try:
        import mlx_whisper  # noqa: F401
        import mlx_audio  # noqa: F401
    except Exception as exc:
        errors.append(f"missing_dependency:{exc}")

    ollama_endpoint = config.ollama.endpoint
    try:
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


def main():
    parser = argparse.ArgumentParser(description="Ollama Vox")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Download recommended Whisper + Kokoro model assets, then exit.",
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config()

    import sys
    from PySide6.QtWidgets import QApplication

    _qt_app = QApplication.instance() or QApplication(sys.argv)

    from ollama_vox.ui.setup_wizard import AppSetupWizard

    if args.setup:
        app_wizard = AppSetupWizard(config)
        app_wizard.run(force_setup=True)
        wizard = OllamaModelWizard(config)
        wizard.run(force_setup=True)
        return

    wizard = OllamaModelWizard(config)
    if not wizard.run():
        sys.exit(1)

    run_startup_health_checks(config)

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
    tts = TTS(
        voice=config.tts.voice,
        rate=config.tts.rate,
        model_id=config.tts.model,
        sample_rate=config.tts.sample_rate,
        split_chars=config.tts.split_chars,
        style_map={
            k: {
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
    pipeline.start()

    app = VoiceApp(pipeline, recorder)
    pipeline.set_status_callback(app.set_status)
    pipeline.set_metrics_callback(app.set_metrics)
    app.run()


if __name__ == "__main__":
    main()
