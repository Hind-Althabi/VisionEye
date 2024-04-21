import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("/Users/hindalthabi/Desktop/Car Counter/streetcount.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("visioneyedetection.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))
center_point = (-10, h)

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            annotator.box_label(box, label=str(track_id), color=colors(int(track_id)))
            annotator.visioneye(box, center_point)

    video_writer.write(im0)
    cv2.imshow("visioneye-pinpoint", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_writer.release()
cap.release()
cv2.destroyAllWindows()