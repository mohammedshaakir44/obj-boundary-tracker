import cv2
import numpy as np

# Open webcam (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# List to store centroid positions for trajectory
trajectory = []

while True:  # Main loop
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    # Morphological closing to connect fragmented edges
    kernel = np.ones((5, 5), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Largest contour
        obj_contour = max(contours, key=cv2.contourArea)

        # Draw contour outline
        cv2.drawContours(frame, [obj_contour], 0, (255, 0, 0), 2)

        # Draw bounding box
        x, y, w, h = cv2.boundingRect(obj_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw convex hull
        hull = cv2.convexHull(obj_contour)
        cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)

        # Compute centroid
        M = cv2.moments(obj_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            trajectory.append((cx, cy))  # Add to trajectory

    # Draw trajectory
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 255, 0), 2)

    # Show the frame
    cv2.imshow("Object Tracker", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
