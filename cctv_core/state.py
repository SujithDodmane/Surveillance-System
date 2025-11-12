# state.py

import threading

#Global variables that can be used by both thread and loop 
faces = []
faces_img = []
faces_img_rgb = []
known_face_embeddings = {}   # load_embeddings from already stored embeddings locally
face_results = {}
latest_frame = None
thread_stop_flag = False
lock = threading.Lock()
key = None
