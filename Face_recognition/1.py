import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---- Register face ----
img = cv2.imread("D:\Projects\Surveillance-System\Face_recognition\RKS_6122 a.JPG")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face = mtcnn(img_rgb)
embedding1 = resnet(face.unsqueeze(0).to(device))

# ---- Compare with another face ----
img2 = cv2.imread("D:\Projects\Surveillance-System\Face_recognition\WIN_20250919_22_08_35_Pro.jpg")
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
face2 = mtcnn(img2_rgb)
embedding2 = resnet(face2.unsqueeze(0).to(device))

# ---- Similarity ----
similarity = F.cosine_similarity(embedding1, embedding2).item()
print("Similarity:", similarity)
