import sys
import os
import numpy as np
from PIL import Image

# Ensure we can import from the root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basemapping.basemapping import BaseMapper

def test_fusion():
    # 1. Create a dummy image
    img = Image.new('RGB', (128, 128), color = (73, 109, 137))
    img.save('fusion_img.png')

    mapper = BaseMapper()

    # Interleaved data
    data = [
        {"mode": "text", "data": "A scene with a blue box."},
        {"mode": "image", "data": "fusion_img.png"},
        {"mode": "text", "data": "The box is still."}
    ]

    bmap = mapper.transform(data, mode="fusion")
    print(f"Fused BaseMap: tokens={len(bmap.tokens)}, matrix shape={bmap.matrix.shape}")

    # Check modality flags (MOD_TEXT=12, MOD_IMAGE=13 in modifier_flags)
    # modifier_flags start at dim 236.
    # So dim 248 is text, 249 is image.
    text_flags = bmap.matrix[0, 236:252]
    image_flags = bmap.matrix[10, 236:252] # Assuming first sentence was < 10 tokens

    print(f"Token 0 modality flags (Text=12): {text_flags[12]}")
    print(f"Token 10 modality flags (Image=13): {image_flags[13]}")

    os.remove('fusion_img.png')

if __name__ == "__main__":
    test_fusion()
