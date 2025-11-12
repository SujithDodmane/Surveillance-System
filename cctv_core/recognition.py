# recognition.py
# Face recognition and embedding comparison threads
# Keeps original logic: crops faces, computes embeddings, compares with known embeddings
# Checks state.key for 'e' and 's' as in your working file

import cv2
import torch
import torch.nn.functional as F
import state
import pickle
import os
from embeddings_manager import register_embedding, save_embeddings


recognizer = None
device = None

def init_models(_recognizer, _device):
    global recognizer, device
    recognizer = _recognizer
    device = _device

class Face_Recognition:
    @staticmethod
    def face_recognition(): 
        global recognizer, device
        while not state.thread_stop_flag:
            if state.latest_frame is not None:
                try:
                    frame_copy = state.latest_frame.copy()
                except Exception:
                    continue
                h, w = frame_copy.shape[:2]
                with state.lock:
                    state.faces_img.clear()
                    state.faces_img_rgb.clear()
                    for i, face in enumerate(state.faces):
                        x1, y1, x2, y2 = face['box']
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            face_cropped = frame_copy[y1:y2, x1:x2]
                            face_cropped_rgb = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB)
                            state.faces_img.append(face_cropped)
                            state.faces_img_rgb.append(face_cropped_rgb)

                            if state.key == ord('e'):
                                # keeping the name or dynamic naming as we want
                                Face_Recognition.register_embeddings(face_cropped_rgb, 'Sujith')
                                print("Face_Embedding_Function called")
                            elif state.key == ord('s'):
                                Face_Recognition.save_embeddings(file_path="face_embeddings_3.pkl")

    #only one face should be there while registering the embeddings
    def register_embeddings(face_rgb, current_name):
        global recognizer, device
        cv2.imshow("Face_Embedded", face_rgb)
        cv2.waitKey(1)

        face_resized = cv2.resize(face_rgb, (160, 160))
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        face_tensor = face_tensor.to(device)
        with torch.no_grad():
            face_embedding = recognizer(face_tensor)

        if current_name not in state.known_face_embeddings:
            state.known_face_embeddings[current_name] = []
        state.known_face_embeddings[current_name].append(face_embedding)
        print(len(state.known_face_embeddings[current_name]))
        pass

    @staticmethod
    def extract_embeddings_compare(): #This has to run every frame
        global recognizer, device
        while not state.thread_stop_flag:
            if state.latest_frame is not None:
                try:
                    frame_copy = state.latest_frame.copy()
                except Exception:
                    continue

                with state.lock:
                    for i, (face_rgb, face) in enumerate(zip(state.faces_img_rgb, state.faces)):
                        try:
                            face_resized = cv2.resize(face_rgb, (160, 160))
                            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                            face_tensor = face_tensor.to(device)
                            with torch.no_grad():
                                face_embedding = recognizer(face_tensor)
                        except Exception:
                            continue

                        best_match = None
                        best_score = -1
                        for name, ref_embedding in state.known_face_embeddings.items():
                            for emb in ref_embedding:
                                if not torch.is_tensor(emb):
                                    emb = torch.tensor(emb, dtype=torch.float32)
                                emb = emb.to(device)
                                face_embedding = face_embedding.to(device)
                                similarity = F.cosine_similarity(emb, face_embedding).item()
                                if similarity > best_score:
                                    best_score = similarity
                                    best_match = name

                        state.face_results[i] = {'name': best_match, 'similarity': best_score}

    def save_embeddings(file_path="face_embeddings_3.pkl"):
        clean_data = {}
        for name, embeds in state.known_face_embeddings.items():
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
