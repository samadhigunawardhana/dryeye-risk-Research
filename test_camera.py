import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # try with AVFOUNDATION
if not cap.isOpened():
    print("Camera NOT opened")
else:
    print("Camera opened OK")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Camera", frame)
        cv2.waitKey(3000)  # show for 3 seconds
    cap.release()
    cv2.destroyAllWindows()
