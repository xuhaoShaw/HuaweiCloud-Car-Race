import cv2

def CapInit():
    cap = cv2.VideoCapture("/dev/video10")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, 0.04)
    cap.set(cv2.CAP_PROP_CONTRAST, 60)
    cap.set(cv2.CAP_PROP_SATURATION, 80)
    return cap




if __name__ == '__main__':
    while True:
        ret, frame = cap.read()
        cv2.imshow('video', frame)
        cv2.waitKey(5)