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

    def _score_emb(self, emb: np.ndarray, target: np.ndarray) -> float:
        """Fast cosine similarity between two vectors (no full pipeline)."""
        a_norm = float(np.linalg.norm(emb))
        b_norm = float(np.linalg.norm(target))
        if a_norm < 1e-10 or b_norm < 1e-10:
            return 0.0
        return float(np.dot(emb / a_norm, target / b_norm))

    def _build_candidate_emb(self, tokens: list) -> np.ndarray:
        """
        Cheap 256-dim candidate vector built purely from base embeddings —
        no pipeline call needed.  Average token base-embeddings, pad to
        FEATURE_DIM with zeros for the modifier dims.
        """
        vecs = []
        for tok in tokens:
            emb = self.mapper._base_vocab.get(tok)
            if emb is None:
                emb = self.mapper._compose_word_embedding(tok)
            vecs.append(emb.astype(np.float32))
        if not vecs:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        avg = np.mean(vecs, axis=0)
        out = np.zeros(FEATURE_DIM, dtype=np.float32)
        out[:len(avg)] = avg
        return out

    def _generative_reconstruction_text(self, target_latent: np.ndarray,
                                         max_tokens: int, iterations: int) -> str:
        """
        Fast greedy text reconstruction using two cheap stages:

          Stage 1: rank ALL vocab words by cosine similarity to the target
                   latent's base region (dims 0:EMBED_DIM).  O(|vocab|).
          Stage 2: for each token position, average the chosen tokens' base
                   embeddings and pick the next candidate that maximises
                   cosine similarity to the full target latent.
                   No full pipeline calls — runs in milliseconds.
        """
        if not self.mapper.is_fitted or not self.mapper._base_vocab:
            return "[unknown]"

        # Stage 1 — cheap pre-ranking using raw base embeddings
        vocab_words = [w for w in self.mapper._base_vocab if " " not in w]
        if not vocab_words:
            vocab_words = list(self.mapper._base_vocab.keys())
        if not vocab_words:
            return "..."

        target_base = target_latent[:EMBED_DIM].astype(np.float32)

        scored: list = []
        for word in vocab_words:
            emb = self.mapper._base_vocab[word].astype(np.float32)
            cos = self._score_emb(emb, target_base)
            scored.append((cos, word))
        scored.sort(key=lambda x: -x[0])

        # Top-K candidates for Stage 2
        k = min(max(iterations, 20), len(scored))
        top_k = [w for _, w in scored[:k]]

        # Stage 2 — greedy token selection using cheap average-embedding score
        current_tokens: list = []
        best_sim: float = -1.0

        for _step in range(max_tokens):
            best_next: str = ""
            improved = False

            for token in top_k:
                cand_emb = self._build_candidate_emb(current_tokens + [token])
                sim = self._score_emb(cand_emb, target_latent)
                if sim > best_sim + 1e-4:
                    best_sim = sim
                    best_next = token
                    improved = True

            if improved and best_next:
                current_tokens.append(best_next)
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
        # Python Fallback for upgraded rendering
        r_val = latent[0] * 127.0 + 128.0
        g_val = latent[1] * 127.0 + 128.0
        b_val = latent[2] * 127.0 + 128.0

        for r in range(128):
            rel_r = r / 128.0
            for c in range(128):
                rel_c = c / 128.0
                val = 0.0
                for i in range(8):
                    freq_r = float(i + 1)
                    freq_c = float(8 - i)
                    phase = latent[16 + i] * np.pi
                    amp = latent[8 + i]
                    val += amp * np.sin(rel_r * 6.28 * freq_r + rel_c * 6.28 * freq_c + phase)

                v = 0.5 + 0.5 * np.tanh(val)
                img_arr[r, c, 0] = np.clip(r_val * v + latent[3] * 50, 0, 255)
                img_arr[r, c, 1] = np.clip(g_val * v + latent[4] * 50, 0, 255)
                img_arr[r, c, 2] = np.clip(b_val * v + latent[5] * 50, 0, 255)

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

    def decode_video(self, latent: np.ndarray, num_frames: int = 10) -> List[Image.Image]:
        """Ultimate SOTA Upgrade: Temporal Continuity Video Generation."""
        lib = _load_lib()
        frames = []
        # Base image from latent
        cur_img = self._decode_image(latent)
        frames.append(cur_img)

        cur_arr = np.array(cur_img).astype(np.uint8)

        # Motion latent derived from the rest of the 256-dim space
        motion = latent.copy()

        for i in range(num_frames - 1):
            next_arr = np.zeros_like(cur_arr)
            if lib:
                lib.render_video_frame_fast(_fp(cur_arr)[0], _fp(motion)[0],
                                          ctypes.c_int(128), ctypes.c_int(128),
                                          ctypes.c_float(0.8), next_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
            else:
                next_arr = cur_arr + (motion[:len(cur_arr.flatten()) % 256] * 5).reshape(cur_arr.shape).astype(np.uint8)

            frames.append(Image.fromarray(next_arr))
            cur_arr = next_arr

        return frames
