from dataclasses import dataclass, field
from typing import Dict, Any, Optional


class ConfigValidationError(Exception):
    pass


def _coerce_type(name: str, value: Any, expected_type: type):
    if value is None:
        return None
    if not isinstance(value, expected_type):
        try:
            return expected_type(value)
        except (ValueError, TypeError):
            raise ConfigValidationError(
                f"Invalid type for '{name}': expected {expected_type.__name__}, got {type(value).__name__}"
            )
    return value


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    vad_enabled: bool = True
    vad_threshold: float = 0.015
    vad_silence_seconds: float = 1.2
    max_duration_seconds: float = 20.0

    def __post_init__(self):
        self.sample_rate = _coerce_type("audio.sample_rate", self.sample_rate, int)
        self.vad_enabled = _coerce_type("audio.vad_enabled", self.vad_enabled, bool)
        self.vad_threshold = _coerce_type(
            "audio.vad_threshold", self.vad_threshold, float
        )
        self.vad_silence_seconds = _coerce_type(
            "audio.vad_silence_seconds", self.vad_silence_seconds, float
        )
        self.max_duration_seconds = _coerce_type(
            "audio.max_duration_seconds", self.max_duration_seconds, float
        )


@dataclass
class STTConfig:
    model: str = "./whisper/whisper-small.en-mlx-q4"

    def __post_init__(self):
        self.model = _coerce_type("stt.model", self.model, str)


@dataclass
class OllamaConfig:
    endpoint: str = "http://localhost:11434"
    model: str = "llama3.2:1b-instruct-q4_K_M"
    temperature: float = 0.7

    def __post_init__(self):
        self.endpoint = _coerce_type("ollama.endpoint", self.endpoint, str)
        self.model = _coerce_type("ollama.model", self.model, str)
        self.temperature = _coerce_type("ollama.temperature", self.temperature, float)


@dataclass
class TTSConfig:
    model: str = "./kokoro/Kokoro-82M-4bit"
    voice: str = "af_bella"
    rate: float = 1.0
    sample_rate: int = 24000
    split_chars: int = 180

    def __post_init__(self):
        self.model = _coerce_type("tts.model", self.model, str)
        self.voice = _coerce_type("tts.voice", self.voice, str)
        self.rate = _coerce_type("tts.rate", self.rate, float)
        self.sample_rate = _coerce_type("tts.sample_rate", self.sample_rate, int)
        self.split_chars = _coerce_type("tts.split_chars", self.split_chars, int)


@dataclass
class HotkeyConfig:
    key: str = "cmd+shift"

    def __post_init__(self):
        self.key = _coerce_type("hotkey.key", self.key, str)


@dataclass
class QueueConfig:
    maxsize: int = 4
    drop_policy: str = "drop_oldest"

    def __post_init__(self):
        self.maxsize = _coerce_type("queue.maxsize", self.maxsize, int)
        self.drop_policy = _coerce_type("queue.drop_policy", self.drop_policy, str)
        if self.drop_policy not in ("drop_oldest", "drop_new", "block"):
            raise ConfigValidationError(
                f"Invalid queue.drop_policy: {self.drop_policy}"
            )


@dataclass
class StyleConfig:
    speed: Optional[float] = None
    pitch: Optional[float] = None
    voice: Optional[str] = None

    def __post_init__(self):
        if self.speed is not None:
            self.speed = _coerce_type("style.speed", self.speed, float)
        if self.pitch is not None:
            self.pitch = _coerce_type("style.pitch", self.pitch, float)
        if self.voice is not None:
            self.voice = _coerce_type("style.voice", self.voice, str)


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    response_style: str = "neutral"
    styles: Dict[str, StyleConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        if not isinstance(data, dict):
            raise ConfigValidationError("Root config must be a dictionary")

        try:
            audio_cfg = AudioConfig(**data.get("audio", {}))
            stt_cfg = STTConfig(**data.get("stt", {}))
            ollama_cfg = OllamaConfig(**data.get("ollama", {}))
            tts_cfg = TTSConfig(**data.get("tts", {}))
            hotkey_cfg = HotkeyConfig(**data.get("hotkey", {}))
            queue_cfg = QueueConfig(**data.get("queue", {}))
        except TypeError as e:
            raise ConfigValidationError(f"Unknown configuration key provided: {e}")

        styles_data = data.get("styles", {})
        if not isinstance(styles_data, dict):
            raise ConfigValidationError("Styles config must be a dictionary")

        try:
            styles = {k: StyleConfig(**v) for k, v in styles_data.items()}
        except TypeError as e:
            raise ConfigValidationError(
                f"Unknown style configuration key provided: {e}"
            )

        return cls(
            audio=audio_cfg,
            stt=stt_cfg,
            ollama=ollama_cfg,
            tts=tts_cfg,
            hotkey=hotkey_cfg,
            queue=queue_cfg,
            response_style=str(data.get("response_style", "neutral")),
            styles=styles,
        )
