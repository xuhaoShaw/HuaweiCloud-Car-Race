from npsocket import NumpySocket
import cv2
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
video_path = BASE_DIR / 'videos/demo.mp4'

cap = cv2.VideoCapture(str(video_path))

sock_sender = NumpySocket()
sock_sender.initialize_sender('127.0.0.1', 9999)

ret, frame = cap.read()

while ret:
    frame = cv2.resize(frame, (620, 480))
    sock_sender.send_array(frame)
    ret, frame = cap.read()

sock_sender.socket.close()
