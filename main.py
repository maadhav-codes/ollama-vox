import yaml
import logging
import os
import requests

from core.audio import AudioRecorder
from core.stt import STT
from core.llm import OllamaClient
from core.tts import TTS
from core.workers import Pipeline
from ui.menubar import VoiceApp


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
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
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
    configure_logging()
    config = load_config()
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
    llm = OllamaClient(config["ollama"]["model"], config["ollama"]["temperature"])
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
    app.run()


if __name__ == "__main__":
    main()
