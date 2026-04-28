import sys
import os

# 1. Obtenemos la ruta actual del nodo y la de la subcarpeta fish_speech
current_dir = os.path.dirname(os.path.realpath(__file__))
fish_speech_dir = os.path.join(current_dir, "fish_speech")

# 2. Las inyectamos en el PATH de Python si no están ya
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if fish_speech_dir not in sys.path:
    sys.path.insert(0, fish_speech_dir)

# Asegurar que existe el archivo .project-root para pyrootutils
project_root_file = os.path.join(fish_speech_dir, ".project-root")
if not os.path.exists(project_root_file):
    with open(project_root_file, "w") as f:
        pass # Crea el archivo vacío si no existe

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