"""
yolo_vehicle.py

Real-time vehicle+person detection + lightweight tracking + accident event detection:
 - vehicle-person collision
 - sudden stop of vehicles
 - falling person / person on road

Usage:
    python yolo_vehicle.py --source 0
    python yolo_vehicle.py --source test.mp4

Requirements:
    pip install ultralytics opencv-python numpy
"""
import cv2
import time
import argparse
import numpy as np
import math
import os
from collections import deque
from ultralytics import YOLO
from ultralytics import YOLO
# ------------------- Helper functions -------------------
def iou_xyxy(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    iw = max(0, xi2 - xi1)
    ih = max(0, yi2 - yi1)
    inter = iw * ih
    area_a = max(1, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(1, (xb2 - xb1) * (yb2 - yb1))
    return inter / (area_a + area_b - inter + 1e-6)

def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def box_area(box):
    x1, y1, x2, y2 = box
    return max(1, (x2 - x1) * (y2 - y1))

def get_head_region(person_box, ratio=0.25):
    x1, y1, x2, y2 = person_box
    head_h = (y2 - y1) * ratio
    return [x1, y1, x2, y1 + head_h]

def save_violation_image(frame, label, obj_id):
    folder = "violations"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{label}_id{obj_id}_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[INFO] Saved violation image: {filename}")

# ------------------- Tracking -------------------
class Track:
    def __init__(self, tid, bbox, cls, max_history=30):
        self.id = tid
        self.bbox = bbox
        self.cls = cls
        self.history = deque(maxlen=max_history)
        self.history.append(bbox)
        self.velocity = np.array([0.0, 0.0])
        self.recent_speeds = deque(maxlen=8)

    def update(self, bbox):
        prev_box = self.bbox
        cprev = box_center(prev_box)
        ccur = box_center(bbox)
        vx, vy = ccur - cprev
        alpha = 0.35
        self.velocity = alpha * np.array([vx, vy]) + (1 - alpha) * self.velocity
        speed = np.linalg.norm(self.velocity)
        self.recent_speeds.append(speed)
        self.bbox = bbox
        self.history.append(bbox)

    def median_speed(self):
        if not self.recent_speeds:
            return 0.0
        return float(np.median(np.array(self.recent_speeds)))

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_missed=8):
        self.next_id = 1
        self.tracks = {}
        self.iou_thr = iou_threshold
        self.max_missed = max_missed
        self.missed = {}

    def update(self, detections):
        matches = {}
        used_tracks = set()
        used_dets = set()

        for d_idx, det in enumerate(detections):
            best_iou = 0.0
            best_tid = None
            for tid, track in self.tracks.items():
                if track.cls != det['cls']:
                    continue
                iouv = iou_xyxy(det['bbox'], track.bbox)
                if iouv > best_iou and iouv >= self.iou_thr:
                    best_iou = iouv
                    best_tid = tid
            if best_tid is not None:
                matches[d_idx] = best_tid
                used_tracks.add(best_tid)
                used_dets.add(d_idx)

        for d_idx, tid in matches.items():
            det = detections[d_idx]
            self.tracks[tid].update(det['bbox'])
            self.missed[tid] = 0

        for d_idx, det in enumerate(detections):
            if d_idx in used_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid, det['bbox'], det['cls'])
            self.missed[tid] = 0

        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.missed[tid] += 1
                if self.missed[tid] > self.max_missed:
                    del self.tracks[tid]
                    del self.missed[tid]

        return list(self.tracks.values())

# ------------------- Accident Detection -------------------
class AccidentDetector:
    def __init__(self):
        self.frame_history = {}
        self.cooldown_seconds = 2.0
        self.last_alert = {}

    def _cooldown_ok(self, key, now):
        last = self.last_alert.get(key, -9999)
        if now - last > self.cooldown_seconds:
            self.last_alert[key] = now
            return True
        return False

    def detect_collisions(self, tracks, now):
        events = []
        for i, t1 in enumerate(tracks):
            for j, t2 in enumerate(tracks):
                if t1.id >= t2.id:
                    continue
                c1 = box_center(t1.bbox)
                c2 = box_center(t2.bbox)
                rel_pos = c2 - c1
                dist = np.linalg.norm(rel_pos)

                rel_vel = t2.velocity - t1.velocity
                closing_speed = np.dot(rel_vel, rel_pos) / (np.linalg.norm(rel_pos)+1e-6)

                size_factor = (math.sqrt(box_area(t1.bbox)) + math.sqrt(box_area(t2.bbox))) / 2
                distance_thresh = max(20, size_factor * 0.5)

                if closing_speed < -5.0 and dist < distance_thresh:
                    key = (t1.id, t2.id)
                    if key not in self.frame_history:
                        self.frame_history[key] = deque(maxlen=3)
                    self.frame_history[key].append(now)
                    if len(self.frame_history[key]) >= 2:
                        if self._cooldown_ok(key, now):
                            events.append({
                                'type': 'collision',
                                'time': now,
                                'id1': t1.id,
                                'id2': t2.id,
                                'dist': dist,
                                'closing_speed': -closing_speed
                            })
                        self.frame_history[key].clear()
                else:
                    key = (t1.id, t2.id)
                    if key in self.frame_history:
                        self.frame_history[key].clear()
        return events

# ------------------- COCO Classes -------------------
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
HELMET_CLASS_ID = 8
COCO_NAMES = {0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck',8:'helmet'}

# ------------------- Main -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--speed_limit", type=float, default=50.0)
    args = parser.parse_args()

    model = YOLO(args.model)
    src = int(args.source) if args.source.isnumeric() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = SimpleTracker(iou_threshold=0.3, max_missed=8)
    helmet_history = {}
    helmet_persist_frames = 3
    accident_detector = AccidentDetector()

    fps_smooth = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]

        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, cl in zip(xyxy, confs, clss):
                if cl not in {PERSON_CLASS_ID, HELMET_CLASS_ID} | VEHICLE_CLASS_IDS:
                    continue
                x1,y1,x2,y2 = map(float, b)
                pad = 0.02 * ((x2-x1)+(y2-y1))
                x1 += pad; y1 += pad; x2 -= pad; y2 -= pad
                detections.append({'bbox':[x1,y1,x2,y2],'cls':int(cl),'conf':float(c)})

        tracks = tracker.update(detections)
        events = accident_detector.detect_collisions(tracks, time.time())

        out = frame.copy()
        y0 = 30

        # Congestion level calculation
        vehicle_count = sum(1 for tr in tracks if tr.cls in VEHICLE_CLASS_IDS)
        if vehicle_count < 20:
            congestion = "Low"
        elif vehicle_count < 50:
            congestion = "Medium"
        else:
            congestion = "High"

        # Draw tracks & alerts
        for tr in tracks:
            x1,y1,x2,y2 = map(int, tr.bbox)
            label = COCO_NAMES.get(tr.cls, str(tr.cls))
            color = (0,255,0) if tr.cls != PERSON_CLASS_ID else (0,120,255)
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            speed = tr.median_speed()

            cv2.putText(out, f"{label} id:{tr.id} s:{speed:.1f}", (x1,max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)

            if speed > args.speed_limit:
                cv2.putText(out, f"OVERSPEED id:{tr.id}", (x1,y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                save_violation_image(frame, "overspeed", tr.id)

                if tr.cls == PERSON_CLASS_ID:
                    head_box = get_head_region(tr.bbox)
                    helmet_present = False
                    for det in detections:
                        if det['cls'] == HELMET_CLASS_ID:
                            if iou_xyxy(head_box, det['bbox']) > 0.5:
                                helmet_present = True
                                break

                    history = helmet_history.get(tr.id, deque(maxlen=helmet_persist_frames))
                    history.append(helmet_present)
                    helmet_history[tr.id] = history
                    helmet_ok = sum(history) > 0

                    if not helmet_ok:
                        cv2.putText(out, f"NO HELMET id:{tr.id}", (x1,y2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                        save_violation_image(frame, "no_helmet", tr.id)

        # Draw collisions
        for e in events:
            cv2.putText(out, "!!! COLLISION DETECTED !!!", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
            print(f"[ALERT] Collision at {time.strftime('%H:%M:%S')} between {e['id1']} & {e['id2']}, dist={e['dist']:.1f}")
            y0 += 30

        # Draw congestion level
        cv2.putText(out, f"Congestion Level: {congestion} ({vehicle_count} vehicles)", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        t1 = time.time()
        fps = 1.0 / (t1-t0) if (t1-t0)>0 else 0
        fps_smooth = fps if fps_smooth is None else 0.9*fps_smooth + 0.1*fps
        cv2.putText(out, f"FPS: {fps_smooth:.2f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        cv2.imshow("Smart Traffic Detection", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
