import cv2
import csv
import os
from ultralytics import YOLO
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
import winsound
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# =========================
# MODEL + TRACKER
# =========================
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.3)

# =========================
# VIDEO INPUT
# =========================
cap = cv2.VideoCapture(0)  # webcam mode

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Width:", width)
print("Height:", height)
print("FPS:", fps)

# =========================
# OUTPUT SETUP
# =========================
os.makedirs("output_videos", exist_ok=True)
os.makedirs("logs", exist_ok=True)

out = cv2.VideoWriter(
    "output_videos/output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# =========================
# CSV SETUP
# =========================
csv_file = open("logs/detections.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Timestamp",
    "Track_ID",
    "Object",
    "Event"
])

# =========================
# REPORT DATA
# =========================
report_data = {
    "total_objects": 0,
    "unique_ids": set(),
    "intrusions": 0
}

seen_tracks = set()
alerted_tracks = set()

# =========================
# ZONE
# =========================
ZONE = (300, 150, 900, 600)

# =========================
# WINDOW
# =========================
cv2.namedWindow("AI Surveillance System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Surveillance System", 1280, 720)

print("Processing started...")

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Draw zone
    cv2.rectangle(frame, (ZONE[0], ZONE[1]), (ZONE[2], ZONE[3]), (0, 0, 255), 2)
    cv2.putText(frame, "RESTRICTED ZONE", (ZONE[0], ZONE[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # YOLO detection
    results = model(frame)[0]

    detections = []

    for box in results.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if confidence > 0.5:
            detections.append((
                [x1, y1, x2 - x1, y2 - y1],
                confidence,
                class_name
            ))

    # TRACKING
    tracks = tracker.update_tracks(detections, frame=frame)

    object_counts = {}

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        class_name = track.det_class

        # Draw box
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id} {class_name}",
                    (int(l), int(t)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # =========================
        # REPORT UPDATE
        # =========================
        report_data["total_objects"] += 1
        report_data["unique_ids"].add(track_id)

        if track_id not in seen_tracks:
            seen_tracks.add(track_id)

            csv_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                track_id,
                class_name,
                "FIRST_SEEN"
            ])

        # =========================
        # ZONE CHECK
        # =========================
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        x1, y1, x2, y2 = ZONE

        in_zone = (x1 < cx < x2) and (y1 < cy < y2)

        if in_zone and track_id not in alerted_tracks:

            alerted_tracks.add(track_id)

            report_data["intrusions"] += 1

            cv2.putText(frame, "ALERT! INTRUSION DETECTED",
                        (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3)

            winsound.Beep(1000, 300)

            csv_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                track_id,
                class_name,
                "ZONE_INTRUSION"
            ])

        # =========================
        # COUNT OBJECTS
        # =========================
        if class_name not in object_counts:
            object_counts[class_name] = set()

        object_counts[class_name].add(track_id)

    # Timestamp
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    # Show counts
    y = 70
    for obj, ids in object_counts.items():
        cv2.putText(frame, f"{obj}: {len(ids)}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2)
        y += 30

    out.write(frame)
    cv2.imshow("AI Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print("Processing Completed Successfully!")

# =========================
# PDF REPORT
# =========================
def generate_pdf():
    c = canvas.Canvas("logs/report.pdf", pagesize=letter)

    c.setFont("Helvetica", 14)
    c.drawString(50, 750, "AI Surveillance System Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"Total Objects Detected: {report_data['total_objects']}")
    c.drawString(50, 680, f"Unique Objects: {len(report_data['unique_ids'])}")
    c.drawString(50, 660, f"Intrusions: {report_data['intrusions']}")

    c.save()

generate_pdf()
print("PDF Report saved in logs/report.pdf")