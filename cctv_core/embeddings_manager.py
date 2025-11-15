# embeddings_manager.py
# Loading, saving and registering embeddings

import os
import pickle
import cv2
import torch

def load_embeddings(file_path="face_embeddings_4.pkl"):
    if not os.path.exists(file_path):
        print("⚠️ No embeddings file found!")
        return {}
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"📂 Embeddings loaded from {file_path}")
    return data

def save_embeddings(known_face_embeddings, file_path="face_embeddings_4.pkl"):
    clean_data = {}
    for name, embeds in known_face_embeddings.items():
        clean_data[name] = []
        for e in embeds:
            if torch.is_tensor(e):
                clean_data[name].append(e.detach().cpu().numpy())
            else:
                clean_data[name].append(e)

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            existing_data = pickle.load(f)
    else:
        existing_data = {}

    for name, embeds in clean_data.items():
        if name not in existing_data:
            existing_data[name] = []
        existing_data[name].extend(embeds)

    with open(file_path, "wb") as f:
        pickle.dump(existing_data, f)
    print(f"💾 Embeddings updated and saved to {file_path}")
