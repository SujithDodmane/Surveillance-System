# 🛡️ Smart CCTV Surveillance System (Face Detection & Recognition)

A **computer vision–based multithreaded CCTV surveillance system** that detects and recognizes faces in real-time using **MTCNN** and **InceptionResnetV1 (FaceNet)**.  
The system ensures smooth real-time performance through multithreading for detection, recognition, and display — ideal for security monitoring, CCTV setups, and automation projects.

---

## 🚀 Features

- Real-time **Face Detection** using MTCNN  
- Face **Recognition** with InceptionResnetV1 (pretrained on VGGFace2)  
- **Multithreading** for smooth parallel detection and display  
- Keyboard Controls:  
  - **E** → Register a new face  
  - **S** → Save embeddings  
  - **Q** → Quit the program safely  
- Displays FPS, confidence, similarity, and recognized names on-screen  
- Stores face embeddings (`.pkl`) for future recognition  

---

## 🧩 Folder Overview

Surveillance-System/
│
├── cctv_core/ # ✅ Main folder containing final working modular code
│ ├── main.py
│ ├── detection.py
│ ├── recognition.py
│ ├── display.py
│ ├── embeddings_manager.py
│ ├── state.py
│ ├── requirements.txt
│ └── README.md # (optional subfolder README)
│
├── .gitignore
├── README.md # (this file)
└── Other folders (legacy/test builds for development reference)

yaml
Copy code

> ⚙️ The **`cctv_core`** folder contains the complete and final working code.  
> Other folders (like `Face_recognition/` or `GUI_enabled/`) are early experiments or testing builds preserved for development history.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/SujithDodmane/Surveillance-System.git
cd Surveillance-System/cctv_core
2️⃣ Create and activate a virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate       # For Windows
# OR
source venv/bin/activate    # For Linux / macOS
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
📦 requirements.txt
Exact versions tested and verified with this project:

ini
Copy code
torch==2.1.2
torchvision==0.16.2
facenet-pytorch==2.5.3
opencv-python==4.10.0.84
numpy==1.26.4
pillow==10.3.0
⚙️ Note:

Works seamlessly on both CPU and GPU (auto-detection).

For GPU acceleration, install CUDA-enabled PyTorch from pytorch.org.

▶️ Running the Program
From inside the cctv_core folder, run:

bash
Copy code
python main.py
The webcam feed will open automatically.
The system will detect and recognize faces in real-time while showing FPS, confidence, and similarity scores.

🔘 Keyboard Controls
Key	Function
E	Register a new face embedding (ensure only one face is visible)
S	Save current embeddings to face_embeddings_3.pkl
Q	Quit the program safely

🧠 Technical Overview
MTCNN: Detects faces and keypoints (eyes, nose, mouth).

InceptionResnetV1 (FaceNet): Generates 512D embeddings for face comparison.

Threading: Four concurrent threads — for detection, recognition, embedding comparison, and display — enable real-time operation.

Embeddings File: Stores recognized faces’ features (.pkl) for persistent use across sessions.

🧪 Tested Environment
Component	Version
Python	3.10
OS	Windows 11 / Linux Ubuntu 22.04
CUDA (optional)	12.1
Torch Device	Auto-detects GPU (CUDA) or CPU

⚡ Performance Tips
Lower the frame resolution (e.g., 960x540) for higher FPS.

Use a GPU for faster embedding computation.

Ensure consistent lighting for accurate detection.

Close background apps using the webcam to prevent frame drops.

🔮 Future Enhancements
🔔 Mobile notifications for unknown face detection.

🌐 Web-based dashboard for remote viewing.

📸 Multi-camera support for large-scale surveillance.

🧾 Attendance & access control integration.

👨‍💻 Author
Sujith D K
B.E. Computer Science and Engineering
R V College of Engineering (RVCE)



📌 Repository Link
🔗 GitHub - SujithDodmane / Surveillance-System
