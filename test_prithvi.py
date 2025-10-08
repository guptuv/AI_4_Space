# # # import torch
# # # from transformers import AutoModel, AutoImageProcessor
# # # from PIL import Image
# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # model_name = "ibm-nasa-geospatial/Prithvi-100M"

# # # processor = AutoImageProcessor.from_pretrained(model_name)
# # # model = AutoModel.from_pretrained(model_name)

# # # print("✅ Model loaded successfully!")

# # # dummy_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
# # # image = Image.fromarray(dummy_image)

# # # inputs = processor(images=image, return_tensors="pt")

# # # with torch.no_grad():
# # #     outputs = model(**inputs)

# # # print("Embedding shape:", outputs.last_hidden_state.shape)

# # # plt.imshow(dummy_image)
# # # plt.title("Dummy Spectral Patch")
# # # plt.show()


# # from transformers import AutoImageProcessor, AutoModelForImageClassification
# # from PIL import Image
# # import torch

# # # Load model and processor
# # model_name = "ibm-nasa-geospatial/Prithvi-100M"
# # processor = AutoImageProcessor.from_pretrained(model_name)
# # model = AutoModelForImageClassification.from_pretrained(model_name)

# # # Load a sample image
# # image_path = "sample_image.jpg"  # Replace with your own image path
# # image = Image.open(image_path).convert("RGB")

# # # Preprocess image
# # inputs = processor(images=image, return_tensors="pt")

# # # Forward pass
# # with torch.no_grad():
# #     outputs = model(**inputs)
# #     logits = outputs.logits
# #     predicted_class_idx = logits.argmax(-1).item()

# # # Print the predicted label
# # label = model.config.id2label.get(predicted_class_idx, f"Label_{predicted_class_idx}")
# # print(f"Predicted class: {label}")

# from transformers import AutoImageProcessor, AutoModel
# from PIL import Image
# import torch

# # Load model and processor
# model_name = "ibm-nasa-geospatial/Prithvi-100M"
# processor = AutoImageProcessor.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # Load a sample image
# image_path = "sample_image.jpg"  # Replace with your image path
# image = Image.open(image_path).convert("RGB")

# # Preprocess image
# inputs = processor(images=image, return_tensors="pt")

# # Get embeddings
# with torch.no_grad():
#     outputs = model(**inputs)
#     last_hidden_state = outputs.last_hidden_state
#     pooled_features = last_hidden_state.mean(dim=1)  # average pooling

# print("✅ Embedding shape:", pooled_features.shape)

from transformers import AutoConfig, AutoModel, AutoImageProcessor

model_name = "ibm-nasa-geospatial/Prithvi-100M"

# --- PATCH: Fix num_labels=None issue ---
config = AutoConfig.from_pretrained(model_name)
if getattr(config, "num_labels", None) is None:
    config.num_labels = 1  # default fallback
# ----------------------------------------

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config)
