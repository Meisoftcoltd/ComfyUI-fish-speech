from .nodes import (
    FishSpeechModelDownloader,
    FishSpeechModelLoader,
    FishSpeechReferenceEncoder,
    FishSpeechTextToSemantic,
    FishSpeechDecoder
)

NODE_CLASS_MAPPINGS = {
    "FishSpeechModelDownloader": FishSpeechModelDownloader,
    "FishSpeechModelLoader": FishSpeechModelLoader,
    "FishSpeechReferenceEncoder": FishSpeechReferenceEncoder,
    "FishSpeechTextToSemantic": FishSpeechTextToSemantic,
    "FishSpeechDecoder": FishSpeechDecoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FishSpeechModelDownloader": "🐟 Download FishSpeech Models",
    "FishSpeechModelLoader": "🐟 FishSpeech Loader",
    "FishSpeechReferenceEncoder": "🐟 Reference Audio Encoder",
    "FishSpeechTextToSemantic": "🐟 Text to Semantic (LLaMA)",
    "FishSpeechDecoder": "🐟 Semantic to Audio (DAC)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']