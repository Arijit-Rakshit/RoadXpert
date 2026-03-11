import cv2
import numpy as np
import datetime
import json

# ---------------- CONFIG ----------------
video_path = "test.mp4"  # Path to your video
pos_line = 250  # Y-position of the counting line
offset = 10     # Line offset
total_count = 0
log_file = "vehicle_log.json"
# ---------------------------------------

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    
    # Background subtraction
    fgmask = fgbg.apply(frame)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicles_people = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1200:  # ignore tiny blobs
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        if aspect_ratio < 0.3 or aspect_ratio > 4.0:
            continue

        cx, cy = x + w // 2, y + h // 2

        # Count when crossing the line
        if (pos_line - offset) < cy < (pos_line + offset):
            total_count += 1
            print(f"🚗/🚶 Object crossed line. Total count: {total_count}")

        vehicles_people.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "center": (int(cx), int(cy))
        })

        # Draw rectangle and center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

    # Congestion level
    congestion_level = (
        "low" if len(vehicles_people) < 3 else
        "medium" if len(vehicles_people) < 7 else
        "high"
    )

    # Display info on frame
    cv2.line(frame, (0, pos_line), (frame.shape[1], pos_line), (255, 0, 0), 2)
    cv2.putText(frame, f"Detected: {len(vehicles_people)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Total count: {total_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Congestion: {congestion_level}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Traffic Detection", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Press Esc to stop
        break

    # Log detections
    if len(vehicles_people) > 0:
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "objects_detected": len(vehicles_people),
            "objects": vehicles_people,
            "total_count": total_count,
            "congestion_level": congestion_level
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

cap.release()
cv2.destroyAllWindows()
print("Processing finished.")
