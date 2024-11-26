import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import warnings

# config
video_path = "data_ext/test_cart_person.mp4"
conf_threshold = 0.5
tracking_classes = [0, 2]  # Theo dõi cả người (0) và xe (2)

# khởi tạo DeepSort
tracker = DeepSort(max_age=5)

# khởi tạo Yolov9
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights="weights/yolov9-c.pt", device=device, fuse=True)
model = AutoShape(model)

# Thêm code xử lý warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# load classname từ file classes.names
with open("data_ext/classes.names") as f:
    class_name = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_name), 3))
tracks = []

# khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(video_path)

# tiến hành đọc từng frame từ video
while True:
    # Đọc
    ret, frame = cap.read()

    if not ret:
        continue

    # Đưa qua model để detect
    results = model(frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_classes is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id not in tracking_classes or confidence < conf_threshold:
                continue
        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Cập nhật gán ID bằng DeepSort
    tracks = tracker.update_tracks(detect, frame=frame)

    # vẽ lên màn hình các khung chữ nhật kèm ID
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = "{}-{}".format(class_name[class_id], track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, R, G), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show hình ảnh lên màn hình
    cv2.imshow("OT", frame)
    # Bấm q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
