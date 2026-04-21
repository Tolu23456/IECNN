import random
import numpy as np
from datasets import load_dataset
from PIL import Image
import os
import pickle

def inject_text_noise(text, level=0.1):
    chars = list(text)
    n = len(chars)
    num_noisy = int(n * level)
    for _ in range(num_noisy):
        idx = random.randint(0, n-1)
        # Random swap or change
        chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz ")
    return "".join(chars)

def inject_image_noise(img_path, level=0.2):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, level * 255, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)
    noisy_path = img_path.replace(".png", "_noisy.png")
    noisy_img.save(noisy_path)
    return noisy_path

def prepare_noisy_data(num_samples=100):
    print(f"Preparing {num_samples} noisy multi-modal samples...")

    # Load 10k Tinystories to fit Global Brain (Noisy Version)
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:10000]", trust_remote_code=True)
    stories = [item['text'] for item in dataset]

    with open("tinystories_noisy_10k.txt", "w", encoding="utf-8") as f:
        for story in stories:
            # 5% noise in 'training' data to test robustness
            noisy_story = inject_text_noise(story, level=0.05)
            f.write(noisy_story.replace('\n', ' ') + "\n")

    # Prepare noisy test samples
    os.makedirs("noisy_multimodal", exist_ok=True)
    noisy_samples = []

    for i in range(num_samples):
        story = stories[i]
        sample_data = []

        # 1. Text with 15% noise
        sample_data.append({"mode": "text", "data": inject_text_noise(story[:100], level=0.15)})

        # 2. Image with 30% noise (random color patch)
        img_path = f"noisy_multimodal/img_{i}.png"
        img = Image.new('RGB', (128, 128), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        img.save(img_path)
        noisy_img_path = inject_image_noise(img_path, level=0.3)
        sample_data.append({"mode": "image", "data": noisy_img_path})

        noisy_samples.append(sample_data)

    with open("noisy_multimodal_100.pkl", "wb") as f:
        pickle.dump(noisy_samples, f)
    print("Noisy dataset prepared.")

if __name__ == "__main__":
    prepare_noisy_data(100)
