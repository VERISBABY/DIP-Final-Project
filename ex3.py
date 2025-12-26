import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("3.mp4")
assert cap.isOpened(), "Error reading video file"

# Define region points
region_points = [(100, 90), (740, 90), (740, 600), (100, 600)]

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("output_ex3.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init trackzone (object tracking in zones, not complete frame)
trackzone = solutions.TrackZone(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="yolo11m.pt",  # use any model that Ultralytics supports, e.g., YOLOv9, YOLOv10
    line_width=1,  # adjust the line width for bounding boxes and text display
    iou=0.45, # IOU threshold
    device="cuda"  # use 'cuda' for GPU inference
)

def preprocess_aerial_frame(frame):
    # Preprocess the frame for better detection
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # resize for better small object detection
    if frame.shape[0] > 1000:  # If video is large
        scale = 0.7
        enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale)
    
    return enhanced

# Process video
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Apply aerial preprocessing
    enhanced = preprocess_aerial_frame(im0)
    results = trackzone(enhanced)
    
    # Get the output frame
    output_frame = results.plot_im
    
    # If the frame was resized in preprocessing, resize it back to original size
    if enhanced.shape[:2] != (h, w):
        output_frame = cv2.resize(output_frame, (w, h))
    
    # Write the video file
    video_writer.write(output_frame)
    
    frame_count += 1
    print(f"Processed frame {frame_count}", end='\r')

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
print(f"Total frames processed: {frame_count}")