# state.py
# Shared global state (mirrors the globals from your single-file script)

import threading

#Global variables that can be used by both thread and loop 
faces = []
faces_img = []
faces_img_rgb = []
known_face_embeddings = {}   # will be initialized by main.py using load_embeddings
face_results = {}
latest_frame = None
thread_stop_flag = False
lock = threading.Lock()
key = None
