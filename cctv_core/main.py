import cv2
import torch
import threading
from facenet_pytorch import MTCNN, InceptionResnetV1
# import time

import state
from embeddings_manager import load_embeddings
from detection import face_detection
from recognition import Face_Recognition, init_models
from display import read_display

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#torch.device('cuda') ---> operation happens on GPU  #torch.device('cpu') ---> operation happens on CPU

detector = MTCNN(keep_all=True, device=device)
recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device=device)

state.known_face_embeddings = load_embeddings()

init_models(recognizer, device)

detection_thread = threading.Thread(target=face_detection, args=(detector,), daemon=True)
read_display_thread = threading.Thread(target=read_display, args=(recognizer, device), daemon=True)
recognition_thread = threading.Thread(target=Face_Recognition.face_recognition, daemon=True)
embedding_compare_thread = threading.Thread(target=Face_Recognition.extract_embeddings_compare, daemon=True)


detection_thread.start()
read_display_thread.start()
recognition_thread.start()
embedding_compare_thread.start()

read_display_thread.join()
cv2.destroyAllWindows()

detection_thread.join()
recognition_thread.join()
embedding_compare_thread.join()
