from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("/Users/hindalthabi/Desktop/Car Counter/streetcount.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# line or region points
line_points = [(10, 310), (520,310)]  
# car, bus, truck classes for count
classes_to_count = [2,5,7]  

# Video writer
video_writer = cv2.VideoWriter("result.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

#Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 region_thickness =2,
                 track_color=(0,255,0)
                 )

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False,
                         classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)
    
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()



