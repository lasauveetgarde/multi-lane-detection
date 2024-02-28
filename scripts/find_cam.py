import cv2
FindCam = []
for i in range(100):
    try:
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            FindCam.append(i)
    except:
        pass
print(str(FindCam))