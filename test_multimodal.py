import sys
import os
import numpy as np
from PIL import Image
import wave
import struct

# Ensure we can import from the root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pipeline import IECNN

def create_dummy_data():
    # 1. Create a dummy image
    img = Image.new('RGB', (128, 128), color = (73, 109, 137))
    img.save('test_image.png')

    # 2. Create a dummy audio file (simple sine wave)
    sample_rate = 22050
    duration = 1.0 # seconds
    frequency = 440.0 # Hz
    n_samples = int(sample_rate * duration)

    with wave.open('test_audio.wav', 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767.0 * 0.5 * np.sin(2.0 * np.pi * frequency * i / sample_rate))
            f.writeframes(struct.pack('<h', value))

def run_multimodal_test():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                  IECNN MULTI-MODAL CONVERGENCE TEST              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    create_dummy_data()
    model = IECNN()

    print("\n[TEST] Running Image Mode...")
    res_img = model.run('test_image.png', mode='image', verbose=True)
    print(f"  → Image Convergence: {res_img.stop_reason} in {res_img.summary['rounds']} rounds")

    print("\n[TEST] Running Audio Mode...")
    res_aud = model.run('test_audio.wav', mode='audio', verbose=True)
    print(f"  → Audio Convergence: {res_aud.stop_reason} in {res_aud.summary['rounds']} rounds")

    print("\n[TEST] Multi-Modal Test Complete.")

    # Cleanup
    os.remove('test_image.png')
    os.remove('test_audio.wav')

if __name__ == "__main__":
    run_multimodal_test()
