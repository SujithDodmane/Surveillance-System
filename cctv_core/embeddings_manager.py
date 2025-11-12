# embeddings_manager.py
# Loading, saving and registering embeddings

import os
import pickle
import cv2
import torch

def load_embeddings(file_path="face_embeddings_3.pkl"):
    if not os.path.exists(file_path):
        print("⚠️ No embeddings file found!")
        return {}
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"📂 Embeddings loaded from {file_path}")
    return data

def register_embedding(face_rgb, current_name, recognizer, device, known_face_embeddings):
    #only one face should be there while registering the embeddings
    cv2.imshow("Face_Embedded", face_rgb)
    cv2.waitKey(1)

    face_resized = cv2.resize(face_rgb, (160, 160))
    face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        face_embedding = recognizer(face_tensor)

    if current_name not in known_face_embeddings:
        known_face_embeddings[current_name] = []
    known_face_embeddings[current_name].append(face_embedding)
    print(f"✅ Face embedding added for {current_name}. Total: {len(known_face_embeddings[current_name])}")

def save_embeddings(known_face_embeddings, file_path="face_embeddings_3.pkl"):
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
