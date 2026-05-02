import argparse
import yaml
import logging
import os
import requests
from huggingface_hub import snapshot_download

try:
    import dateutil.parser
    import phonemizer
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if not hasattr(EspeakWrapper, "set_data_path"):

        def _set_data_path(path):
            import os

            os.environ["ESPEAK_DATA_PATH"] = str(path)

        EspeakWrapper.set_data_path = _set_data_path
    import misaki.espeak
except ImportError:
    pass

from core.audio import AudioRecorder
from core.stt import STT
from core.llm import OllamaClient
from core.tts import TTS
from core.workers import Pipeline
from ui.tray_app import VoiceTrayApp as VoiceApp


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
    )


def run_startup_health_checks(config):
    errors = []
    tts_model = config.get("tts", {}).get("model")
    stt_model = config.get("stt", {}).get("model")
    if tts_model and not os.path.exists(tts_model):
        errors.append(f"missing_tts_model:{tts_model}")
    if stt_model and not os.path.exists(stt_model):
        errors.append(f"missing_stt_model:{stt_model}")
    try:
        import mlx_whisper  # noqa: F401
        import mlx_audio  # noqa: F401
    except Exception as exc:
        errors.append(f"missing_dependency:{exc}")

    ollama_endpoint = config.get("ollama", {}).get("endpoint", "http://localhost:11434")
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


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def run_setup(config):
    logger = logging.getLogger(__name__)
    stt_path = config.get("stt", {}).get("model", "./whisper/whisper-small.en-mlx-q4")
    tts_path = config.get("tts", {}).get("model", "./kokoro/Kokoro-82M-4bit")
    voice = config.get("tts", {}).get("voice", "af_bella")

    logger.info(
        "event=setup_started stt_path=%s tts_path=%s voice=%s",
        stt_path,
        tts_path,
        voice,
    )

    if os.path.exists(stt_path):
        logger.info("event=setup_skip_stt reason=exists path=%s", stt_path)
    else:
        _ensure_parent(stt_path)
        logger.info(
            "event=setup_download_stt repo=mlx-community/whisper-small.en-mlx-q4"
        )
        snapshot_download(
            repo_id="mlx-community/whisper-small.en-mlx-q4",
            local_dir=stt_path,
            local_dir_use_symlinks=False,
        )
        logger.info("event=setup_done_stt path=%s", stt_path)

    if os.path.exists(tts_path):
        logger.info("event=setup_skip_tts reason=exists path=%s", tts_path)
    else:
        _ensure_parent(tts_path)
        logger.info("event=setup_download_tts repo=mlx-community/Kokoro-82M-4bit")
        allow_patterns = [
            "*.json",
            "*.md",
            "*.safetensors",
            "*.pth",
            f"voices/{voice}.safetensors",
            f"voices/{voice}.pt",
        ]
        snapshot_download(
            repo_id="mlx-community/Kokoro-82M-4bit",
            local_dir=tts_path,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
        logger.info("event=setup_done_tts path=%s", tts_path)

    logger.info("event=setup_finished")


def main():
    parser = argparse.ArgumentParser(description="Native Ollama Voiceover")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Download recommended Whisper + Kokoro model assets, then exit.",
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config()

    if args.setup:
        run_setup(config)
        return

    run_startup_health_checks(config)

    audio_cfg = config.get("audio", {})
    recorder = AudioRecorder(
        sample_rate=audio_cfg.get("sample_rate", 16000),
        vad_enabled=audio_cfg.get("vad_enabled", True),
        vad_threshold=audio_cfg.get("vad_threshold", 0.015),
        vad_silence_seconds=audio_cfg.get("vad_silence_seconds", 1.2),
        max_duration_seconds=audio_cfg.get("max_duration_seconds", 20.0),
    )

    stt = STT(config["stt"]["model"])
    llm = OllamaClient(
        endpoint=config["ollama"].get("endpoint", "http://localhost:11434"),
        model=config["ollama"]["model"],
        temperature=config["ollama"]["temperature"],
    )
    tts = TTS(
        voice=config["tts"]["voice"],
        rate=config["tts"]["rate"],
        model_id=config["tts"].get("model"),
        sample_rate=config["tts"].get("sample_rate", 24000),
        split_chars=config["tts"].get("split_chars", 180),
        style_map=config.get("styles", {}),
    )

    queue_cfg = config.get("queue", {})
    pipeline = Pipeline(
        stt,
        llm,
        tts,
        audio_cfg.get("sample_rate", 16000),
        queue_maxsize=queue_cfg.get("maxsize", 4),
        drop_policy=queue_cfg.get("drop_policy", "drop_oldest"),
        response_style=config.get("response_style", "neutral"),
    )
    pipeline.start()

    hotkey = config.get("hotkey", {}).get("key", "cmd+shift")
    app = VoiceApp(pipeline, recorder, hotkey=hotkey)
    pipeline.set_status_callback(app.set_status)
    pipeline.set_metrics_callback(app.set_metrics)
    app.run()


if __name__ == "__main__":
    main()
