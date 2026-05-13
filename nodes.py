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

import folder_paths

# Importing fish_speech modules
from fish_speech.models.text2semantic.inference import init_model as init_llama_model, generate_long
from fish_speech.models.dac.inference import load_model as load_dac_model
from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora


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
    CATEGORY = "🐟 FishSpeech/Audio"

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
    CATEGORY = "🐟 FishSpeech/Utils"

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
    CATEGORY = "🐟 FishSpeech/Loaders"

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
    CATEGORY = "🐟 FishSpeech/Audio"

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
                "max_seq_len": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 32768,
                    "step": 256,
                    "display": "number"
                }),
                "max_new_tokens": ("INT", {"default": 4096, "min": 128, "max": 8192}),

                # Ajustado a 200 para mitigar el Speaker Drift según el manual
                "chunk_length": ("INT", {"default": 200, "min": 50, "max": 4096, "step": 10}),

                # Desbloqueamos 2 decimales y fijamos los defaults óptimos
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.10, "max": 2.00, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.80, "min": 0.10, "max": 1.00, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.10, "min": 0.50, "max": 2.00, "step": 0.01}),
            },
            "optional": {
                "prompt_tokens": ("FS_PROMPT_TOKENS",),
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("FS_SEMANTIC_TOKENS",)
    RETURN_NAMES = ("semantic_tokens",)
    FUNCTION = "generate_semantic"
    CATEGORY = "🐟 FishSpeech/Generation"

    def generate_semantic(self, llama_model, text, max_seq_len, max_new_tokens, chunk_length, temperature, top_p, repetition_penalty, prompt_tokens=None, prompt_text=""):
        import gc
        print("Generando tokens semánticos a partir del texto...")

        # Inyección automática de Anclaje de Identidad
        clean_text = text.strip()
        if not clean_text.startswith("<|speaker:0|>"):
            text = f"<|speaker:0|> {clean_text}"
            print("⚓ Anclaje de identidad <|speaker:0|> inyectado automáticamente.")

        model = llama_model["model"]
        decode_one_token = llama_model["decode_one_token"]
        device = llama_model["device"]

        # 🚀 VRAM JUGGLING INICIO: Forzar modelo a GPU antes de trabajar
        model.to(device)

        model.config.max_seq_len = max_seq_len

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=max_seq_len,
                dtype=next(model.parameters()).dtype,
            )

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

        if not codes:
            semantic_tokens = torch.empty((0,), device="cpu")
        else:
            # Mandamos los tokens directamente a CPU para no ocupar VRAM
            semantic_tokens = torch.cat(codes, dim=1).cpu()

        # 🧹 VRAM JUGGLING FIN: Expulsar modelo a RAM (CPU) y vaciar caché de la gráfica
        print("🧹 Liberando 15GB de VRAM del modelo LLaMA...")
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        return (semantic_tokens,)

class FishSpeechDecoder:
    """Decodifica los tokens semánticos a una forma de onda acústica usando el DAC y normaliza el volumen."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder_model": ("FS_DECODER_MODEL",),
                "semantic_tokens": ("FS_SEMANTIC_TOKENS",),
                "normalize_audio": ("BOOLEAN", {"default": True}),
                "target_peak_db": ("FLOAT", {"default": -1.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode_audio"
    CATEGORY = "🐟 FishSpeech/Generation"

    def decode_audio(self, decoder_model, semantic_tokens, normalize_audio, target_peak_db):
        import gc
        print("Decodificando tokens a forma de onda de audio...")

        # 🚀 VRAM JUGGLING INICIO: Subir decoder a GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder_model.to(device)

        indices = semantic_tokens.to(device)

        if indices.ndim == 2:
            indices = indices.unsqueeze(0)

        with torch.no_grad():
            fake_audios = decoder_model.from_indices(indices)

        waveform = fake_audios.cpu()

        # Motor de Normalización de Volumen
        if normalize_audio:
            print(f"🔊 Normalizando volumen al pico de {target_peak_db} dB...")
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                target_linear = 10 ** (target_peak_db / 20)
                waveform = waveform * (target_linear / max_val)

        audio_output = {"waveform": waveform, "sample_rate": decoder_model.sample_rate}

        # 🧹 VRAM JUGGLING FIN: Expulsar decoder a RAM y limpiar
        print("🧹 Liberando VRAM del decodificador DAC...")
        decoder_model.to("cpu")
        del indices
        gc.collect()
        torch.cuda.empty_cache()

        return (audio_output,)

class FishSpeechLoraLoader:
    """Carga y aplica un LoRA al modelo LLaMA de Fish Speech."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llama_model": ("FS_LLAMA_MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "r": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "alpha": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("FS_LLAMA_MODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "apply_lora"
    CATEGORY = "🐟 FishSpeech/Loaders"

    def apply_lora(self, llama_model, lora_name, r, alpha):
        print(f"Configurando arquitectura LoRA (r={r}, alpha={alpha})...")
        try:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                raise FileNotFoundError(f"Archivo LoRA no encontrado: {lora_name}")

            model = llama_model["model"]
            device = llama_model["device"]

            # 1. Configurar la estructura LoRA EXACTAMENTE igual que en el entrenamiento
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=0.0,
                target_modules=["attention", "mlp", "embeddings", "output"]
            )
            setup_lora(model, lora_config)

            print(f"Cargando pesos del LoRA: {lora_name}...")
            if lora_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                lora_state_dict = load_file(lora_path)
            else:
                lora_state_dict = torch.load(lora_path, map_location="cpu", weights_only=False)
                if "state_dict" in lora_state_dict:
                    lora_state_dict = lora_state_dict["state_dict"]

            # 2. ESCUDO: Extraer SOLO los pesos LoRA y alinear el dtype
            cleaned_state_dict = {}
            target_dtype = next(model.parameters()).dtype

            for k, v in lora_state_dict.items():
                if "lora" in k:  # <-- Filtro vital para no sobrescribir el modelo base
                    clean_key = k.replace("model.", "", 1) if k.startswith("model.") else k
                    cleaned_state_dict[clean_key] = v.to(dtype=target_dtype)

            # 3. Cargar los pesos limpios en el modelo y registrar diagnóstico
            missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

            print(f"🔥 LoRA inyectado. Total de capas LoRA cargadas: {len(cleaned_state_dict)}")
            if len(unexpected) > 0:
                print(f"⚠️ Aviso: {len(unexpected)} llaves del LoRA no encontraron destino.")

            model.to(device)
            print("LoRA aplicado correctamente.")
            return (llama_model,)
        except (RuntimeError, FileNotFoundError, Exception) as e:
            print(f"Error al cargar el LoRA: {str(e)}")
            raise RuntimeError(f"Error al cargar el LoRA {lora_name}: {str(e)}")
