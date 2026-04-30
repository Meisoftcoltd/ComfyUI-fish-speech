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

# 3. AHORA SÍ, hacemos el resto de importaciones
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel

# Importing fish_speech modules
from fish_speech.models.text2semantic.inference import init_model as init_llama_model, generate_long
from fish_speech.models.dac.inference import load_model as load_dac_model


class FishSpeechWhisperTranscriber:
    """Transcribe el audio de referencia a texto usando faster-whisper, optimizado contra alucinaciones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_size": (["tiny", "base", "small", "medium", "large-v3"], {"default": "base"}),
                "language": (["auto", "es", "en", "fr", "de", "it", "pt", "ja", "zh"], {"default": "auto"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_text",)
    FUNCTION = "transcribe"
    CATEGORY = "FishSpeech/Audio"

    def transcribe(self, audio, model_size, language, device):
        print(f"Cargando modelo Whisper ({model_size}) en {device}...")
        compute_type = "float16" if device == "cuda" else "int8"

        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                print("Error: Memoria VRAM insuficiente para cargar Whisper. Por favor, intenta usar el modelo 'base' o 'small', o cambia el dispositivo a 'cpu'.")
                raise RuntimeError("Error de VRAM: Memoria insuficiente para el modelo Whisper seleccionado. Usa uno más pequeño o CPU.") from e
            else:
                raise e

        # Extraer tensor y sample rate del formato estándar de ComfyUI [batch, channels, samples]
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Convertir a mono si es estéreo (promedio de canales)
        # Evaluamos explicitamente si hay mas de 1 canal en la dimension correcta (1 usualmente para batch, channels, samples)
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        # Resamplear a 16000Hz (requerido por Whisper)
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convertir a numpy array 1D (colapsando cualquier otra dimension de batch/channels a 1D plano)
        audio_np = waveform.flatten().numpy()

        print("Transcribiendo audio (con VAD anti-alucinaciones)...")
        lang_param = None if language == "auto" else language

        try:
            segments, info = model.transcribe(
                audio_np,
                language=lang_param,
                beam_size=5,
                vad_filter=True, # CRÍTICO: Elimina silencios para evitar alucinaciones
                condition_on_previous_text=False # CRÍTICO: Evita bucles de repetición
            )

            # Unir todos los segmentos detectados
            transcription = " ".join([segment.text.strip() for segment in segments])
            print(f"Transcripción detectada ({info.language}): {transcription}")

            return (transcription,)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                print("Error: Memoria VRAM insuficiente durante la transcripción de Whisper. Por favor, intenta usar un modelo más pequeño o CPU.")
                raise RuntimeError("Error de VRAM: Memoria insuficiente durante la transcripción. Usa un modelo Whisper más pequeño o CPU.") from e
            else:
                raise e

class FishSpeechModelDownloader:
    """Nodo extra para descargar los modelos desde HuggingFace directamente al directorio de ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo": (["fishaudio/openaudio-s1-mini", "fishaudio/s2-pro"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("checkpoint_path",)
    FUNCTION = "download_model"
    CATEGORY = "FishSpeech/Utils"

    def download_model(self, model_repo):
        # Descarga el modelo a models/fish_speech
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models/fish_speech")
        save_path = os.path.join(base_path, model_repo.split("/")[-1])

        print(f"Descargando modelo {model_repo} a {save_path}...")
        snapshot_download(repo_id=model_repo, local_dir=save_path)
        print("Descarga completada.")

        return (save_path,)

class FishSpeechModelLoader:
    """Carga los pesos de LLaMA (Texto a Semántica) y el Decoder (DAC)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {"default": "models/fish_speech/s2-pro"}),
                "decoder_config": (["modded_dac_vq"],),
                "llama_device": (["cuda", "cpu"], {"default": "cuda"}),
                "decoder_device": (["cuda", "cpu"], {"default": "cpu"}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("FS_LLAMA_MODEL", "FS_DECODER_MODEL")
    RETURN_NAMES = ("llama_model", "decoder_model")
    FUNCTION = "load_models"
    CATEGORY = "FishSpeech/Loaders"

    def load_models(self, checkpoint_path, decoder_config, llama_device, decoder_device, precision):
        print("Cargando LLaMA y Codec de Fish Speech...")

        precision_dtype = torch.bfloat16
        if precision == "float16":
            precision_dtype = torch.float16
        elif precision == "float32":
            precision_dtype = torch.float32

        # Initialize LLaMA model
        llama_model, decode_one_token = init_llama_model(
            checkpoint_path=checkpoint_path,
            device=llama_device,
            precision=precision_dtype,
            compile=False
        )
        llama_wrapper = {
            "model": llama_model,
            "decode_one_token": decode_one_token,
            "device": llama_device
        }

        # Initialize DAC Decoder model
        codec_path = os.path.join(checkpoint_path, "codec.pth")
        if not os.path.exists(codec_path):
             # Let's check for firefly
            if os.path.exists(os.path.join(checkpoint_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")):
                 codec_path = os.path.join(checkpoint_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")

        decoder_model = load_dac_model(
            config_name=decoder_config,
            checkpoint_path=codec_path,
            device=decoder_device
        )

        return (llama_wrapper, decoder_model)

class FishSpeechReferenceEncoder:
    """Procesa un audio de referencia para extraer los 'prompt_tokens' (fake.npy) para clonación de voz."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder_model": ("FS_DECODER_MODEL",),
                "audio": ("AUDIO",), # Formato de audio estándar de ComfyUI
            }
        }

    RETURN_TYPES = ("FS_PROMPT_TOKENS",)
    RETURN_NAMES = ("prompt_tokens",)
    FUNCTION = "encode_reference"
    CATEGORY = "FishSpeech/Audio"

    def encode_reference(self, decoder_model, audio):
        print("Extrayendo tokens del audio de referencia...")

        # Audio from ComfyUI is typically a dict: {"waveform": tensor(B, C, T), "sample_rate": int}
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        device = next(decoder_model.parameters()).device

        # If stereo, take mean to get mono
        if waveform.shape[1] > 1:
            waveform = waveform.mean(1, keepdim=True)

        # Resample to the decoder's expected sample rate
        waveform = torchaudio.functional.resample(waveform, sample_rate, decoder_model.sample_rate)
        waveform = waveform.to(device)

        # Obtain VQ Tokens from the DAC Encoder
        audio_lengths = torch.tensor([waveform.shape[2]], device=device, dtype=torch.long)

        with torch.no_grad():
            indices, _ = decoder_model.encode(waveform, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0] # Take first batch

        prompt_tokens = indices # Keep it as tensor
        return (prompt_tokens,)

class FishSpeechTextToSemantic:
    """Toma el texto (y opcionalmente tokens de referencia) y genera los tokens semánticos."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llama_model": ("FS_LLAMA_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hola mundo, probando Fish Speech."}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 128, "max": 8192}),
                "chunk_length": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
            },
            "optional": {
                "prompt_tokens": ("FS_PROMPT_TOKENS",),
                "prompt_text": ("STRING", {"multiline": True, "default": ""}), # Texto transcrito del audio de referencia
            }
        }

    RETURN_TYPES = ("FS_SEMANTIC_TOKENS",)
    RETURN_NAMES = ("semantic_tokens",)
    FUNCTION = "generate_semantic"
    CATEGORY = "FishSpeech/Generation"

    def generate_semantic(self, llama_model, text, max_new_tokens, chunk_length, temperature, top_p, repetition_penalty, prompt_tokens=None, prompt_text=""):
        print("Generando tokens semánticos a partir del texto...")

        model = llama_model["model"]
        decode_one_token = llama_model["decode_one_token"]
        device = llama_model["device"]

        # Configure caches for generation
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )

        # Call generate_long which takes care of splitting text, applying prompts, and iterating chunks
        generator = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=False,
            iterative_prompt=True,
            chunk_length=chunk_length,
            prompt_text=prompt_text if prompt_text else None,
            prompt_tokens=prompt_tokens if prompt_tokens is not None else None,
        )

        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)

        # Concatenate generated codes
        if not codes:
            semantic_tokens = torch.empty((0,), device=device)
        else:
            semantic_tokens = torch.cat(codes, dim=1)

        return (semantic_tokens,)

class FishSpeechDecoder:
    """Decodifica los tokens semánticos a una forma de onda acústica usando el DAC."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder_model": ("FS_DECODER_MODEL",),
                "semantic_tokens": ("FS_SEMANTIC_TOKENS",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode_audio"
    CATEGORY = "FishSpeech/Generation"

    def decode_audio(self, decoder_model, semantic_tokens):
        print("Decodificando tokens a forma de onda de audio...")

        device = next(decoder_model.parameters()).device
        indices = semantic_tokens.to(device)

        if indices.ndim == 2:
            indices = indices.unsqueeze(0)

        with torch.no_grad():
            fake_audios = decoder_model.from_indices(indices)

        # Structure for ComfyUI audio node: {"waveform": tensor(B, C, T), "sample_rate": int}
        waveform = fake_audios.cpu()

        audio_output = {"waveform": waveform, "sample_rate": decoder_model.sample_rate}
        return (audio_output,)
