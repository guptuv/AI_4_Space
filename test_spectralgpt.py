import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Model name (open-access)
model_name = "ibm-nasa-geospatial/Prithvi-100M"

# Load processor and model
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("✅ Model loaded successfully!")

# ----------------------------
# Example: test on a small image
# ----------------------------
# Create a dummy "satellite patch" (RGB 64x64)
dummy_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
image = Image.fromarray(dummy_image)

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

print("Embedding shape:", outputs.last_hidden_state.shape)

# Optional: visualize dummy image
plt.imshow(dummy_image)
plt.title("Dummy Spectral Patch")
plt.show()
