import random
import numpy as np
from datasets import load_dataset
from PIL import Image
import os
import pickle
import math
import wave
import struct

def inject_extreme_text_noise(text, level=0.5):
    chars = list(text)
    n = len(chars)
    num_noisy = int(n * level)
    for _ in range(num_noisy):
        idx = random.randint(0, n-1)
        chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz !?,.")
    return "".join(chars)

def inject_extreme_image_noise(img_path, level=0.5):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32)
    # High frequency noise + dropout
    noise = np.random.normal(0, level * 255, arr.shape)
    dropout = np.random.choice([0, 1], size=arr.shape, p=[level, 1-level])
    noisy_arr = np.clip(arr * dropout + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)
    noisy_path = img_path.replace(".png", "_extreme.png")
    noisy_img.save(noisy_path)
    return noisy_path

def generate_tone(filepath, freq=440, duration=0.5, sr=22050):
    n_samples = int(sr * duration)
    with wave.open(filepath, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        for i in range(n_samples):
            value = int(32767.0 * math.sin(2.0 * math.pi * freq * i / sr))
            f.writeframes(struct.pack('<h', value))

def inject_extreme_audio_noise(aud_path, level=0.5):
    with wave.open(aud_path, 'rb') as f:
        params = f.getparams()
        frames = f.readframes(params.nframes)

    samples = list(struct.unpack('<' + ('h' * params.nframes), frames))
    num_noisy = int(len(samples) * level)
    for _ in range(num_noisy):
        idx = random.randint(0, len(samples)-1)
        samples[idx] = random.randint(-32768, 32767)

    noisy_path = aud_path.replace(".wav", "_extreme.wav")
    with wave.open(noisy_path, 'wb') as f:
        f.setparams(params)
        f.writeframes(struct.pack('<' + ('h' * len(samples)), *samples))
    return noisy_path

def prepare_extreme_dataset(num_samples=100000):
    print(f"Loading {num_samples} samples from TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{num_samples}]", trust_remote_code=True)
    stories = [item['text'] for item in dataset]

    print("Injecting 50% noise into corpus...")
    with open("tinystories_extreme_100k.txt", "w", encoding="utf-8") as f:
        for i, story in enumerate(stories):
            if i % 10000 == 0: print(f"  Processed {i}...")
            noisy = inject_extreme_text_noise(story, level=0.5)
            f.write(noisy.replace('\n', ' ') + "\n")

    # Test set (100 multi-modal samples with 50% noise)
    os.makedirs("extreme_multimodal", exist_ok=True)
    extreme_samples = []
    for i in range(100):
        story = stories[i]
        sample = []
        # Text
        sample.append({"mode": "text", "data": inject_extreme_text_noise(story[:100], level=0.5)})
        # Image
        img_path = f"extreme_multimodal/img_{i}.png"
        Image.new('RGB', (128, 128), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))).save(img_path)
        sample.append({"mode": "image", "data": inject_extreme_image_noise(img_path)})
        # Audio
        aud_path = f"extreme_multimodal/aud_{i}.wav"
        generate_tone(aud_path, freq=random.randint(200, 800))
        sample.append({"mode": "audio", "data": inject_extreme_audio_noise(aud_path)})

        extreme_samples.append(sample)

    with open("extreme_multimodal_100.pkl", "wb") as f:
        pickle.dump(extreme_samples, f)
    print("Extreme dataset (100k) prepared.")

if __name__ == "__main__":
    prepare_extreme_dataset(100000)
