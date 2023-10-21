"""
This script processes video streams to detect and highlight black objects.
Adapted and improved upon based on the tutorial from the official OpenCV documentation.
Source: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
"""

import cv2 as cv
import numpy as np

# Initialize the video capture using the default camera (camera index 0)
cap = cv.VideoCapture(0)

# Create a kernel for morphological operations, specifically a 5x5 matrix of ones
kernel = np.ones((5, 5), np.uint8)

# Start the main loop to process each video frame
while True:
    # Capture the frame from the video stream
    _, frame = cap.read()

    # Convert the BGR frame to the HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the lower and upper boundaries for the color "black" in the HSV space
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    # Threshold the HSV frame to isolate the black color
    mask = cv.inRange(hsv, lower_black, upper_black)

    # Apply a Gaussian blur to the mask to reduce noise
    blurred = cv.GaussianBlur(mask, (5, 5), 0)

    # Apply a morphological opening operation (erosion followed by dilation) to further reduce noise
    mask_cleaned = cv.morphologyEx(blurred, cv.MORPH_OPEN, kernel)

    # Find the contours in the cleaned mask
    contours, _ = cv.findContours(mask_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if contours:
        # Get the largest contour based on area
        largest_contour = max(contours, key=cv.contourArea)

        # Compute the minimum area rectangle that encloses the largest contour
        rect = cv.minAreaRect(largest_contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        
        # Draw the minimum area rectangle on the original frame, and mask
        cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
        cv.drawContours(mask_cleaned, [box], 0, (0, 255, 0), 2)

    # Display the original frame with the drawn contour
    cv.imshow('frame', frame)
    # Display the cleaned mask
    cv.imshow('mask', mask_cleaned)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv.waitKey(1) & 0xFF == ord('0'):
        break

# Release the video capture object
cap.release()
# Close all OpenCV windows
cv.destroyAllWindows()

