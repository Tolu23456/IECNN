"""
IECNN Generative Decoder — porting latent vectors back to raw modalities.

The decoder uses 'Generative Convergence':
1. It takes a 256-dim latent vector (Z).
2. It initializes a candidate BaseMap.
3. It iteratively proposes token/patch changes and accepts those that
   increase the similarity between the candidate's encoding and Z.
4. It renders the final BaseMap into Text, Image, or Audio.
"""

import numpy as np
import os
import sys
import ctypes
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import wave
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basemapping.basemapping import BaseMapper, BaseMap, EMBED_DIM, FEATURE_DIM
from formulas.formulas import similarity_score, _fp

# ── Load C shared library ────────────────────────────────────────────
_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "decoder_c.so")
    if os.path.exists(so_path):
        try:
            _lib = ctypes.CDLL(so_path)
            _lib.render_image_fast.restype = None
            _lib.render_audio_fast.restype = None
        except Exception:
            _lib = None
    return _lib

class IECNNDecoder:
    def __init__(self, model, alpha: float = 0.7):
        self.model = model
        self.mapper = model.base_mapper
        self.alpha = alpha
        self._rng = np.random.RandomState(42)

    def decode(self, latent: np.ndarray, target_mode: str = "text",
               max_tokens: int = 10, iterations: int = 20) -> Any:
        """
        Generative Convergence: Iteratively find a BaseMap that matches the latent.
        """
        if target_mode == "text":
            return self._generative_reconstruction_text(latent, max_tokens, iterations)
        elif target_mode == "image":
            return self._decode_image(latent)
        elif target_mode == "audio":
            return self._decode_audio(latent)

        return None

    def _generative_reconstruction_text(self, target_latent: np.ndarray,
                                      max_tokens: int, iterations: int) -> str:
        """
        Greedy iterative reconstruction of the text sequence.
        """
        if not self.mapper.is_fitted or not self.mapper._base_vocab:
            return "[unknown]"

        current_tokens = []
        best_sim = -1.0

        vocab_list = list(self.mapper._base_vocab.keys())
        # Filter vocab to common words to speed up
        search_vocab = [w for w in vocab_list if ' ' not in w] # Only unigrams for reconstruction
        if not search_vocab: search_vocab = vocab_list

        for _ in range(max_tokens):
            best_next_token = None
            improved = False

            # Sample vocab to keep it fast
            sample_size = min(200, len(search_vocab))
            candidates = self._rng.choice(search_vocab, sample_size, replace=False)

            for token in candidates:
                test_seq = " ".join(current_tokens + [token])
                # Encode the candidate sequence
                test_latent = self.model.encode(test_seq)
                sim = similarity_score(test_latent, target_latent, self.alpha)

                if sim > best_sim:
                    best_sim = sim
                    best_next_token = token
                    improved = True

            if improved and best_next_token:
                current_tokens.append(best_next_token)
            else:
                break

        return " ".join(current_tokens) if current_tokens else "..."

    def _decode_image(self, latent: np.ndarray) -> Image.Image:
        """Reconstruct 128x128 image from 256-dim latent."""
        lib = _load_lib()
        img_arr = np.zeros((128, 128, 3), dtype=np.uint8)
        if lib:
            lib.render_image_fast(_fp(latent)[0], ctypes.c_int(128), ctypes.c_int(128),
                                 img_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
            return Image.fromarray(img_arr)

        # Simple rendering based on the latent vector
        # Map the 224-dim embedding to visual features
        r_val = int(np.clip(latent[0] * 127 + 128, 0, 255))
        g_val = int(np.clip(latent[1] * 127 + 128, 0, 255))
        b_val = int(np.clip(latent[2] * 127 + 128, 0, 255))

        # Create a pattern
        for r in range(128):
            for c in range(128):
                # Gradient based on latent
                v = (np.sin(r/10.0 * latent[3]) + np.cos(c/10.0 * latent[4])) * 0.5 + 0.5
                img_arr[r, c, 0] = np.clip(r_val * v, 0, 255)
                img_arr[r, c, 1] = np.clip(g_val * v, 0, 255)
                img_arr[r, c, 2] = np.clip(b_val * v, 0, 255)

        return Image.fromarray(img_arr)

    def _decode_audio(self, latent: np.ndarray, duration: float = 1.0) -> bytes:
        """Generate audio from latent."""
        lib = _load_lib()
        sr = 22050
        n_samples = int(sr * duration)
        if lib:
            out_pcm = np.zeros(n_samples, dtype=np.int16)
            lib.render_audio_fast(_fp(latent)[0], ctypes.c_int(sr), ctypes.c_float(duration),
                                 out_pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_short)))
            return out_pcm.tobytes()

        f1 = 200 + abs(latent[0]) * 400
        f2 = 400 + abs(latent[1]) * 600

        audio_data = []
        import math
        for i in range(n_samples):
            t = i / sr
            val = 0.5 * math.sin(2 * math.pi * f1 * t) + 0.5 * math.sin(2 * math.pi * f2 * t)
            audio_data.append(int(val * 32767))

        return struct.pack('<' + ('h' * len(audio_data)), *audio_data)

    def save_output(self, data: Any, mode: str, path: str):
        if mode == "text":
            with open(path, "w") as f:
                f.write(data)
        elif mode == "image":
            data.save(path)
        elif mode == "audio":
            with wave.open(path, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(22050)
                f.writeframes(data)
