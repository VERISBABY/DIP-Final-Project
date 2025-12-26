import cv2
import numpy as np
from ultralytics import YOLO
import time

class DistanceCalculator:
    def __init__(self, video_path, model_path="yolo11n.pt"):
        self.video_path = video_path
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        print(f"Model loaded: {model_path}")
        
        # Mouse interaction state
        self.selected_points = []  # Will store (x, y) coordinates
        self.distance_log = []
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file: {video_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video loaded: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Create output video writer
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path = f"distance_calculation_output_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        # Create window and set mouse callback
        self.window_name = "Distance Calculator"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Processing state
        self.processing = True
        self.current_detections = []
    
    def mouse_callback(self, event, x, y, flags, param):
        # Handle mouse events for selecting objects

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is on a detection
            selected_detection = self.find_closest_detection(x, y)
            
            if selected_detection is not None:
                # Add detection's center point
                self.selected_points.append(selected_detection)
                print(f"Selected object at ({selected_detection[0]:.0f}, {selected_detection[1]:.0f})")
                
                # If we have 2 points, calculate distance
                if len(self.selected_points) == 2:
                    self.calculate_distance()
            else:
                print("Click on a detected object (bounding box)")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clear all selected points
            self.selected_points.clear()
            print("Cleared all selected points")
    
    def find_closest_detection(self, x, y, threshold=30):
        
        # find the closest detection to the mouse click, returns the center point (x, y) if close enough
        
        if not self.current_detections:
            return None
        
        min_distance = float('inf')
        closest_point = None
        
        for detection in self.current_detections:
            center_x, center_y = detection
            distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)
            
            if distance < threshold and distance < min_distance:
                min_distance = distance
                closest_point = (center_x, center_y)
        
        return closest_point
    
    def calculate_distance(self):
        # Calculate distance between two selected points
        if len(self.selected_points) != 2:
            return
        
        point1 = self.selected_points[0]
        point2 = self.selected_points[1]
        
        # Calculate Euclidean distance
        distance_pixels = np.sqrt(
            (point2[0] - point1[0])**2 + 
            (point2[1] - point1[1])**2
        )
        
        # Store in log
        distance_info = {
            'point1': point1,
            'point2': point2,
            'distance_pixels': distance_pixels,
            'frame': self.frame_count
        }
        # print(distance_info)
        
        print(f"Distance calculated: {distance_pixels:.2f} pixels")
        print(f"Between: ({point1[0]:.0f}, {point1[1]:.0f}) and ({point2[0]:.0f}, {point2[1]:.0f})")
    
    def process_video(self):
        #Main video processing loop
        self.frame_count = 0
        
        print("\n" + "="*5)
        print("DISTANCE CALCULATION APPLICATION")
        print("="*60)
        print("Instructions:")
        print("1. LEFT click on TWO detected objects to measure distance")
        print("2. Distance line will be drawn automatically")
        print("3. RIGHT click to clear selections")
        print("4. Press 'q' to quit processing")
        print("5. Press 's' to show current distances")
        print("="*60)
        
        while self.processing and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                print("\nEnd of video reached")
                break
            
            self.frame_count += 1
            
            # Run YOLO detection
            results = self.model(frame, conf=0.3, verbose=False)
            
            # Get detections
            processed_frame = frame.copy()
            self.current_detections = []
            
            # Process YOLO results
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Draw bounding boxes and collect centers
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = map(int, box[:4])
                    class_id = int(cls)
                    class_name = self.model.names[class_id]
                    
                    # Calculate center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Store for mouse interaction
                    self.current_detections.append((center_x, center_y))
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label with class and confidence
                    label = f"{class_name} {conf:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(processed_frame, 
                                (x1, y1 - label_height - 5),
                                (x1 + label_width, y1),
                                color, -1)
                    cv2.putText(processed_frame, label, 
                              (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw selected points and distance line
            for i, (x, y) in enumerate(self.selected_points):
                # Draw selected point
                point_color = (255, 0, 0) if i == 0 else (0, 0, 255)  # Blue for point1, Red for point2
                cv2.circle(processed_frame, (int(x), int(y)), 8, point_color, -1)
                cv2.putText(processed_frame, f"P{i+1}", 
                          (int(x) + 10, int(y) - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, point_color, 2)
            
            # Draw line and distance if 2 points selected
            if len(self.selected_points) == 2:
                point1 = self.selected_points[0]
                point2 = self.selected_points[1]
                
                # Draw line between points
                line_color = (0, 255, 255)  # Yellow
                cv2.line(processed_frame, 
                        (int(point1[0]), int(point1[1])), 
                        (int(point2[0]), int(point2[1])), 
                        line_color, 2)
                
                # Calculate and display distance
                distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                
                # Calculate midpoint for text placement
                mid_x = int((point1[0] + point2[0]) / 2)
                mid_y = int((point1[1] + point2[1]) / 2)
                
                # Display distance
                distance_text = f"Distance: {distance:.1f} px"
                cv2.putText(processed_frame, distance_text,
                          (mid_x - 80, mid_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
            
            # Add frame counter and instructions
            cv2.putText(processed_frame, f"Frame: {self.frame_count}/{self.total_frames}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Selected: {len(self.selected_points)}/2", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show distances measured
            cv2.putText(processed_frame, f"Measurements: {len(self.distance_log)}", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Instructions : "Left click: Select object | Right click: Clear | Q: Quit"
            
            # Display the frame
            cv2.imshow(self.window_name, processed_frame)
            
            # Write frame to output video
            self.out.write(processed_frame)
            
            # Update progress every 50 frames
            if self.frame_count % 50 == 0:
                print(f"Processing frame {self.frame_count}/{self.total_frames}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nStopped by user")
                self.processing = False
                break
            elif key == ord('s'):
                self.show_distance_summary()
            elif key == ord('c'):
                self.selected_points.clear()
                print("Cleared selected points")
        
        # Cleanup
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
    
    def show_distance_summary(self):
        #Show all measured distances
        print("\n" + "="*60)
        print("DISTANCE MEASUREMENTS SUMMARY")
        print("="*60)
        
        if not self.distance_log:
            print("No distances measured yet.")
        else:
            for i, dist in enumerate(self.distance_log):
                print(f"\nMeasurement {i+1}:")
                print(f"  Frame: {dist['frame']}")
                print(f"  Point 1: ({dist['point1'][0]:.0f}, {dist['point1'][1]:.0f})")
                print(f"  Point 2: ({dist['point2'][0]:.0f}, {dist['point2'][1]:.0f})")
                print(f"  Distance: {dist['distance_pixels']:.2f} pixels")
        
        print("="*60)

if __name__ == "__main__":
    VIDEO_PATH = "2.mp4" 
    
    try:
        # Initialize and run distance calculator
        calculator = DistanceCalculator(
            video_path=VIDEO_PATH,
            model_path="yolo11n.pt"
        )
        
        calculator.process_video()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Current directory files:")
        import os
        for file in os.listdir('.'):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"  - {file}")
    except Exception as e:
        print(f"Error: {e}")