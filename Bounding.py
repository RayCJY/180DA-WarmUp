import numpy as np
import cv2

cap = cv2.VideoCapture(0)
lower_black_hsv = np.array([0, 0, 0])
upper_black_hsv = np.array([180, 255, 30])

kernel = np.ones((5, 5), np.uint8)

while True:
    _, frame = cap.read()
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, lower_black_hsv, upper_black_hsv)

    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    mask_cleaned = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        cv2.drawContours(mask_cleaned, [box], 0, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('blackThreshold', mask_cleaned)

    if cv2.waitKey(1) & 0xFF == ord('0'):
        break

cap.release()
cv2.destroyAllWindows()
