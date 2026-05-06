"""Application configuration module for Ollama Vox.

This module defines the entire configuration schema for the application using
Python ``dataclasses``. Each dataclass represents one logical section of the
``config.yaml`` file (e.g. ``[audio]``, ``[stt]``, ``[ollama]``, etc.).

Why dataclasses?
    Dataclasses give us a clean, self-documenting way to define typed
    configuration fields with default values — no external libraries needed.
    They also support ``__post_init__``, which we use to validate and coerce
    YAML values to the correct Python type right after object creation.

Typical usage example:
    >>> import yaml
    >>> from ollama_vox.core.config import AppConfig
    >>> with open("config.yaml") as f:
    ...     data = yaml.safe_load(f)
    >>> config = AppConfig.from_dict(data)
    >>> print(config.audio.sample_rate)
    16000
"""

from dataclasses import dataclass, field
from typing import Any


class ConfigValidationError(Exception):
    """Raised when a configuration value is invalid or of the wrong type.

    This custom exception class is used throughout the config module to
    signal that something in ``config.yaml`` doesn't match what the
    application expects. It inherits directly from ``Exception`` so it can
    be caught explicitly by the caller (e.g. in ``main.py``).

    Example:
        >>> raise ConfigValidationError("audio.sample_rate must be an int")
        ConfigValidationError: audio.sample_rate must be an int
    """

    pass


def _coerce_type(name: str, value: Any, expected_type: type):
    """Attempt to cast a raw config value to the required Python type.

    YAML is a loosely-typed format. A user may write ``sample_rate: "16000"``
    (a string) instead of ``sample_rate: 16000`` (an integer). This helper
    tries to silently convert such values. If conversion is impossible it
    raises :class:`ConfigValidationError` with a helpful message.

    Special handling for ``bool``:
        Python's built-in ``bool("False")`` returns ``True`` (because any
        non-empty string is truthy). We therefore treat booleans as a special
        case and accept the common English words ``"true"``, ``"false"``,
        ``"yes"``, ``"no"``, ``"1"``, ``"0"``, ``"t"``, ``"f"``, ``"y"``,
        and ``"n"`` — all case-insensitive.

    Args:
        name (str): The dotted config key, e.g. ``"audio.sample_rate"``.
                    Used only in error messages so the user knows which field
                    caused the problem.
        value (Any): The raw value read from the YAML file.
        expected_type (type): The Python type we need, e.g. ``int``,
                              ``float``, ``str``, or ``bool``.

    Returns:
        Any: The value cast to ``expected_type``, or ``None`` if ``value``
             was ``None``.

    Raises:
        ConfigValidationError: If the value cannot be cast to the expected
            type. For example, passing ``"hello"`` when ``int`` is expected.

    Example:
        >>> _coerce_type("audio.sample_rate", "16000", int)
        16000
        >>> _coerce_type("audio.vad_enabled", "yes", bool)
        True
        >>> _coerce_type("audio.sample_rate", "not_a_number", int)
        ConfigValidationError: Invalid type for 'audio.sample_rate': ...
    """
    # None is a valid sentinel that means "use the default" — pass it through.
    if value is None:
        return None

    # If the value is already the right type, no conversion needed.
    if not isinstance(value, expected_type):
        try:
            # Bool coercion requires special logic because Python's bool()
            # considers *any* non-empty string to be True, which would make
            # bool("False") == True — clearly wrong for config files.
            if expected_type is bool and isinstance(value, str):
                v_lower = value.lower().strip()
                if v_lower in ("true", "1", "t", "y", "yes"):
                    return True
                elif v_lower in ("false", "0", "f", "n", "no"):
                    return False
                else:
                    raise ValueError(f"Cannot coerce '{value}' to bool")

            # For all other types (int, float, str) just call the constructor.
            return expected_type(value)
        except (ValueError, TypeError):
            raise ConfigValidationError(
                f"Invalid type for '{name}': expected {expected_type.__name__}, got {type(value).__name__}"
            )

    return value


@dataclass
class AudioConfig:
    """Configuration for the microphone and Voice Activity Detection (VAD).

    Controls how audio is captured from the user's microphone and how the
    application decides when the user has finished speaking (VAD = Voice
    Activity Detection, a technique that automatically detects silence to
    stop recording).

    Attributes:
        sample_rate (int): Number of audio samples captured per second.
            16000 Hz (16 kHz) is the standard for speech recognition models
            like Whisper. Changing this without also updating the STT model
            may degrade transcription accuracy. Default: 16000.
        vad_enabled (bool): When ``True``, the recorder monitors the audio
            energy level and stops automatically once it detects a sustained
            period of silence (see ``vad_silence_seconds``).
            When ``False``, recording only stops when the user clicks Stop
            or the ``max_duration_seconds`` limit is reached.
            Default: True.
        vad_threshold (float): The Root-Mean-Square (RMS) energy level below
            which audio is considered "silent". Values are in the range
            [0.0, 1.0] for normalised 32-bit float audio. A lower value
            means the recorder is *more sensitive* to silence; a higher value
            means it requires louder noise before it considers the mic active.
            Default: 0.015.
        vad_silence_seconds (float): How many consecutive seconds of silence
            must pass before the recorder auto-stops. For example, a value
            of 1.2 means: "if the mic has been quiet for at least 1.2 seconds
            in a row, assume the user has finished speaking."
            Default: 1.2.
        max_duration_seconds (float): Hard upper limit on recording length in
            seconds. Even if the user keeps speaking, recording stops after
            this many seconds. This prevents very long audio clips that could
            overwhelm the STT model. Default: 20.0.

    Example YAML section::

        audio:
          sample_rate: 16000
          vad_enabled: true
          vad_threshold: 0.015
          vad_silence_seconds: 1.2
          max_duration_seconds: 20.0
    """

    sample_rate: int = 16000
    vad_enabled: bool = True
    vad_threshold: float = 0.015
    vad_silence_seconds: float = 1.2
    max_duration_seconds: float = 20.0

    def __post_init__(self):
        """Validate and coerce all fields to their correct types after init.

        Python's ``@dataclass`` calls this method automatically right after
        ``__init__`` completes. We use it to ensure that even if the user
        wrote a number as a string in YAML (e.g. ``"16000"`` instead of
        ``16000``), the final Python object always has the right type.

        Raises:
            ConfigValidationError: If any field value cannot be converted to
                its expected type.
        """
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
    """Configuration for the Speech-to-Text (STT) engine.

    Specifies which Whisper model the application should load for
    transcribing the user's speech into text. The model must be a
    locally downloaded MLX-format Whisper checkpoint.

    Attributes:
        model (str): File-system path to the Whisper model directory.
            This should point to a directory containing the MLX-quantised
            model files (e.g. ``weights.npz``, ``config.json``).
            Default: ``"./whisper/whisper-small.en-mlx-q4"`` — a small,
            fast English-only model quantised to 4-bit precision.

    Example YAML section::

        stt:
          model: ./whisper/whisper-small.en-mlx-q4
    """

    model: str = "./whisper/whisper-small.en-mlx-q4"

    def __post_init__(self):
        """Coerce the model path to a string.

        Raises:
            ConfigValidationError: If the value cannot be converted to str.
        """
        self.model = _coerce_type("stt.model", self.model, str)


@dataclass
class OllamaConfig:
    """Configuration for the Ollama large-language-model (LLM) server.

    Ollama is a local server that runs LLMs on the user's machine. This
    dataclass stores the URL where the server is listening, which model to
    use, and how "creative" the model's responses should be.

    Attributes:
        endpoint (str): Base URL of the running Ollama server.
            Must include the scheme (``http://`` or ``https://``) but should
            *not* include a trailing slash. Default: ``"http://localhost:11434"``.
        model (str): The Ollama model tag to use for chat completions.
            Must match a model that has already been downloaded via
            ``ollama pull <model>``. Example values: ``"llama3.2:1b-instruct-q4_K_M"``,
            ``"mistral"``, ``"phi3"``.
            Default: ``"llama3.2:1b-instruct-q4_K_M"``.
        temperature (float): Sampling temperature passed to the model.
            Controls how random/creative the output is.
            * 0.0 → deterministic, always picks the most likely next token.
            * 1.0 → very creative / random.
            * Values around 0.7 are a good default for conversational use.
            Default: 0.7.

    Example YAML section::

        ollama:
          endpoint: http://localhost:11434
          model: llama3.2:1b-instruct-q4_K_M
          temperature: 0.7
    """

    endpoint: str = "http://localhost:11434"
    model: str = "llama3.2:1b-instruct-q4_K_M"
    temperature: float = 0.7

    def __post_init__(self):
        """Validate and coerce all fields to their correct Python types.

        Raises:
            ConfigValidationError: If any field value is of an incompatible type.
        """
        self.endpoint = _coerce_type("ollama.endpoint", self.endpoint, str)
        self.model = _coerce_type("ollama.model", self.model, str)
        self.temperature = _coerce_type("ollama.temperature", self.temperature, float)


@dataclass
class TTSConfig:
    """Configuration for the Text-to-Speech (TTS) synthesis engine.

    Controls the Kokoro-based TTS model that converts the LLM's text response
    back into spoken audio that is played through the speakers.

    Attributes:
        model (str): File-system path to the Kokoro model directory.
            Should contain the MLX model weights (``*.safetensors``) and
            config files. Default: ``"./kokoro/Kokoro-82M-4bit"``.
        voice (str): Voice identifier used for synthesis. The voice name
            corresponds to a ``.safetensors`` or ``.pt`` file inside the
            model's ``voices/`` subdirectory.
            Example: ``"af_bella"``, ``"af_sky"``.
            Default: ``"af_bella"``.
        rate (float): Global speech rate multiplier applied to all synthesis
            unless overridden by a style. 1.0 = normal speed, 0.8 = slower,
            1.3 = faster. Default: 1.0.
        sample_rate (int): Audio sample rate (Hz) used for the TTS output.
            Kokoro produces audio at 24 kHz, so this should generally stay
            at 24000 to avoid resampling artefacts. Default: 24000.
        split_chars (int): Maximum number of characters in a single TTS
            synthesis chunk. Long texts are split at sentence boundaries and
            then hard-split at this length to keep synthesis latency low.
            Smaller values → lower latency, more audio segments.
            Default: 180.

    Example YAML section::

        tts:
          model: ./kokoro/Kokoro-82M-4bit
          voice: af_bella
          rate: 1.0
          sample_rate: 24000
          split_chars: 180
    """

    model: str = "./kokoro/Kokoro-82M-4bit"
    voice: str = "af_bella"
    rate: float = 1.0
    sample_rate: int = 24000
    split_chars: int = 180

    def __post_init__(self):
        """Validate and coerce all TTS fields to their correct Python types.

        Raises:
            ConfigValidationError: If any field value cannot be converted.
        """
        self.model = _coerce_type("tts.model", self.model, str)
        self.voice = _coerce_type("tts.voice", self.voice, str)
        self.rate = _coerce_type("tts.rate", self.rate, float)
        self.sample_rate = _coerce_type("tts.sample_rate", self.sample_rate, int)
        self.split_chars = _coerce_type("tts.split_chars", self.split_chars, int)


@dataclass
class QueueConfig:
    """Configuration for the internal audio/text processing queues.

    The pipeline uses three bounded ``queue.Queue`` instances to hand data
    between the STT, LLM, and TTS worker threads. When a queue is full and
    a new item arrives, the ``drop_policy`` decides what to do.

    Attributes:
        maxsize (int): Maximum number of items each queue can hold. A smaller
            value reduces memory usage and keeps latency low; a larger value
            provides a bigger buffer during momentary slowdowns.
            Default: 4.
        drop_policy (str): What to do when a queue is full and a new item
            arrives. Valid values:
            * ``"drop_oldest"`` — remove the oldest item in the queue to
              make room for the new one. Best when you always want the
              freshest user input to be processed.
            * ``"drop_new"`` — discard the incoming item and keep the queue
              unchanged. Best when you don't want to lose already-queued work.
            * ``"block"`` — wait until a slot is available (blocking the
              producing thread). Use carefully — this can cause the UI to
              feel unresponsive.
            Default: ``"drop_oldest"``.

    Raises:
        ConfigValidationError: If ``drop_policy`` is not one of the three
            valid values listed above.

    Example YAML section::

        queue:
          maxsize: 4
          drop_policy: drop_oldest
    """

    maxsize: int = 4
    drop_policy: str = "drop_oldest"

    def __post_init__(self):
        """Validate the queue settings after initialisation.

        Raises:
            ConfigValidationError: If ``drop_policy`` is not a recognised value.
        """
        self.maxsize = _coerce_type("queue.maxsize", self.maxsize, int)
        self.drop_policy = _coerce_type("queue.drop_policy", self.drop_policy, str)

        # Reject unknown policies early so the error is clear at startup.
        if self.drop_policy not in ("drop_oldest", "drop_new", "block"):
            raise ConfigValidationError(
                f"Invalid queue.drop_policy: {self.drop_policy}"
            )


@dataclass
class StyleConfig:
    """Configuration for a named TTS speaking style.

    Styles allow the user to define named presets for how the TTS engine
    speaks. For example, a ``"friendly"`` style might use a higher-pitched
    voice at a slightly faster rate, while a ``"neutral"`` style uses the
    defaults. The active style is selected via ``AppConfig.response_style``
    and applied in :class:`ollama_vox.core.tts.TTS`.

    All attributes are optional (``None`` means "use the TTS default").

    Attributes:
        speed (Optional[float]): Speech rate multiplier for this style.
            Overrides the global ``tts.rate`` when this style is active.
            Example: ``1.3`` for 30% faster speech. Default: ``None``.
        pitch (Optional[float]): Pitch shift multiplier. Whether the TTS
            model supports this depends on the underlying ``mlx-audio``
            implementation; if unsupported, it is silently ignored.
            Example: ``1.1`` for slightly higher pitch. Default: ``None``.
        voice (Optional[str]): Voice ID to use for this style, overriding
            the global ``tts.voice``. Default: ``None``.

    Example YAML section::

        styles:
          friendly:
            speed: 1.2
            pitch: 1.1
            voice: af_sky
          neutral:
            speed: 1.0
    """

    speed: float | None = None
    pitch: float | None = None
    voice: str | None = None

    def __post_init__(self):
        """Coerce non-None style fields to their correct types.

        Only coerces fields that have been explicitly set (i.e. are not None),
        so that ``None`` keeps its meaning of "use the default".

        Raises:
            ConfigValidationError: If a provided value cannot be converted.
        """
        if self.speed is not None:
            self.speed = _coerce_type("style.speed", self.speed, float)
        if self.pitch is not None:
            self.pitch = _coerce_type("style.pitch", self.pitch, float)
        if self.voice is not None:
            self.voice = _coerce_type("style.voice", self.voice, str)


@dataclass
class AppConfig:
    """Root configuration object that aggregates all sub-configs.

    This is the single object passed around the application that contains
    every configuration setting. It is created once at startup by calling
    :meth:`from_dict` with the parsed YAML data, and then passed to all
    subsystems (audio recorder, STT, LLM, TTS, pipeline, UI).

    Attributes:
        audio (AudioConfig): Microphone and VAD settings.
        stt (STTConfig): Speech-to-Text (Whisper) settings.
        ollama (OllamaConfig): Ollama LLM server settings.
        tts (TTSConfig): Text-to-Speech (Kokoro) settings.
        queue (QueueConfig): Pipeline queue settings.
        response_style (str): The name of the active speaking style from the
            ``styles`` dictionary. Must match a key in ``styles`` (or be
            ``"neutral"`` to use no special style). Default: ``"neutral"``.
        styles (Dict[str, StyleConfig]): A mapping of style name → style
            settings, parsed from the ``styles:`` section of ``config.yaml``.
            Default: empty dict (no custom styles).

    Example:
        >>> config = AppConfig.from_dict({"audio": {"sample_rate": 8000}})
        >>> config.audio.sample_rate
        8000
        >>> config.ollama.model  # falls back to default
        'llama3.2:1b-instruct-q4_K_M'
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

    queue: QueueConfig = field(default_factory=QueueConfig)
    response_style: str = "neutral"
    styles: dict[str, StyleConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Parse and validate a raw dictionary (from YAML) into an AppConfig.

        This is the main factory method for creating an ``AppConfig``. It
        takes the dictionary produced by ``yaml.safe_load()`` and constructs
        all sub-configs from the appropriate nested sections. Missing sections
        are replaced with defaults.

        Args:
            data (dict): A dictionary with optional keys ``"audio"``,
                ``"stt"``, ``"ollama"``, ``"tts"``, ``"queue"``,
                ``"response_style"``, and ``"styles"``. Each key's value
                should itself be a dict of field-name → value pairs.
                Unknown keys inside a section raise a
                :class:`ConfigValidationError`.

        Returns:
            AppConfig: A fully validated and type-coerced configuration object.

        Raises:
            ConfigValidationError: If:
                * ``data`` is not a dictionary.
                * Any known section contains an unrecognised key.
                * Any field value cannot be coerced to its expected type.
                * ``queue.drop_policy`` is not one of the allowed strings.
                * The ``styles`` section is not a dictionary.

        Example:
            >>> AppConfig.from_dict({})  # all defaults
            AppConfig(audio=AudioConfig(...), ...)
            >>> AppConfig.from_dict({"ollama": {"model": "mistral"}}).ollama.model
            'mistral'
        """
        # Guard against malformed YAML (e.g. a list at the root level).
        if not isinstance(data, dict):
            raise ConfigValidationError("Root config must be a dictionary")

        try:
            # Build each sub-config from its YAML section, or use defaults
            # if the section is absent (data.get(...) returns {}).
            audio_cfg = AudioConfig(**data.get("audio", {}))
            stt_cfg = STTConfig(**data.get("stt", {}))
            ollama_cfg = OllamaConfig(**data.get("ollama", {}))
            tts_cfg = TTSConfig(**data.get("tts", {}))

            queue_cfg = QueueConfig(**data.get("queue", {}))
        except TypeError as e:
            # dataclasses raise TypeError for unexpected keyword arguments.
            # We re-raise with a friendlier message.
            raise ConfigValidationError(f"Unknown configuration key provided: {e}")

        # Validate that the styles section is a dict before iterating over it.
        styles_data = data.get("styles", {})
        if not isinstance(styles_data, dict):
            raise ConfigValidationError("Styles config must be a dictionary")

        try:
            # Build a StyleConfig object for each named style.
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
            queue=queue_cfg,
            # Default to "neutral" if not specified; always convert to str.
            response_style=str(data.get("response_style", "neutral")),
            styles=styles,
        )
