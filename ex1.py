# run by 'streamlit run ex1.py --server.fileWatcherType=none'
import os
import streamlit as st
import cv2
from ultralytics import YOLO
import supervision as sv
import time
from datetime import datetime

st.set_page_config(
    page_title="People Detection with Enhanced YOLO",
    page_icon="👥",
    layout="wide"
)

st.title("People Detection with YOLO11 and Computer Vision Libraries")

# PRE-DEFINED CONFIGURATION
VIDEO_PATH = "1.mp4"
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.35
OUTPUT_FILENAME = f"detected_people_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

#  SESSION STATE INITIALIZATION
def init_session_state():
    defaults = {
        'processing': False,
        'paused': False,
        'frame_count': 0,
        'total_people': 0,
        'max_people': 0,
        'current_fps': 0,
        'start_time': None,
        'cap': None,
        'video_writer': None,
        'output_path': None,
        'detector': None,
        'last_frame': None,
        'people_in_current_frame': 0,
        'total_video_frames': 0,
        'last_display_time': 0,
        'display_interval': 0.033,
        'processing_complete': False,
        'final_stats': None,
        'video_placeholder': None,
        'auto_started': False,  # Track if auto-start has been performed
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# DETECTOR Class w Multi-library Integration
class PeopleDetector:
    def __init__(self, model_name=MODEL_NAME, conf_threshold=CONFIDENCE_THRESHOLD):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Try to import supervision
        self.supervision_available = False
        try:
            self.sv = sv
            self.tracker = sv.ByteTrack()
            self.box_annotator = sv.BoxAnnotator(thickness=2, color=sv.ColorPalette.DEFAULT)
            self.label_annotator = sv.LabelAnnotator(
                text_position=sv.Position.TOP_LEFT,
                text_scale=0.5,
                text_thickness=1,
                text_color=sv.Color.BLACK
            )
            self.supervision_available = True
        except (ImportError, AttributeError):
            self.tracker = None
            self.box_annotator = None
            self.label_annotator = None
    
    def preprocess_frame(self, frame):
        if frame is None:
            return None
        
        processed = frame.copy()
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)
        enhanced_lab = cv2.merge((l_channel, a, b))
        processed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        return processed
    
    def detect_people(self, frame):
        #Integrated detection pipeline
        if frame is None:
            return frame, 0, []
        
        enhanced_frame = self.preprocess_frame(frame)
        results = self.model(
            enhanced_frame,
            conf=self.conf_threshold,
            classes=[0],
            verbose=False,
            iou=0.45
        )
        
        annotated_frame = frame.copy()
        people_count = 0
        tracker_ids = []
        
        if len(results) > 0 and results[0].boxes is not None:
            people_count = len(results[0].boxes)
            
            if self.supervision_available and people_count > 0:
                try:
                    detections = self.sv.Detections.from_ultralytics(results[0])
                    detections = self.tracker.update_with_detections(detections)
                    
                    labels = []
                    for _, confidence, _, tracker_id in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
                        if tracker_id is not None:
                            label = f"Person {tracker_id}"
                            tracker_ids.append(tracker_id)
                        else:
                            label = f"Person"
                        labels.append(label)
                    
                    annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                except Exception:
                    annotated_frame = results[0].plot()
            else:
                annotated_frame = results[0].plot()
        
        return annotated_frame, people_count, tracker_ids

def auto_start_processing():
    # Automatically start processing when app loads
    if not st.session_state.auto_started and os.path.exists(VIDEO_PATH):
        st.session_state.processing = True
        st.session_state.paused = False
        st.session_state.start_time = time.time()
        st.session_state.processing_complete = False
        st.session_state.final_stats = None
        st.session_state.frame_count = 0
        st.session_state.total_people = 0
        st.session_state.max_people = 0
        st.session_state.auto_started = True
        return True
    return False

# STREAMLIT UI
# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    if os.path.exists(VIDEO_PATH):
        cap_test = cv2.VideoCapture(VIDEO_PATH)
        fps = int(cap_test.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap_test.release()
        
        st.info(f"**Video:** `{VIDEO_PATH}`\n\n**Frames:** {total_frames}\n**FPS:** {fps}\n**Duration:** {duration:.1f}s")
    else:
        st.error(f"Video not found: {VIDEO_PATH}")
    
    st.header("Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", type="primary", disabled=st.session_state.processing or not os.path.exists(VIDEO_PATH)):
            st.session_state.processing = True
            st.session_state.paused = False
            st.session_state.start_time = time.time()
            st.session_state.processing_complete = False
            st.session_state.final_stats = None
            st.session_state.frame_count = 0
            st.session_state.total_people = 0
            st.session_state.max_people = 0
            st.session_state.auto_started = True
            st.rerun()
    with col2:
        if st.button("⏸️ Pause", disabled=not st.session_state.processing):
            st.session_state.paused = not st.session_state.paused
            st.rerun()
    
    if st.button("🛑 Stop", disabled=not st.session_state.processing):
        st.session_state.processing = False
        st.session_state.paused = False
        st.rerun()

if st.session_state.detector is None:
    st.session_state.detector = PeopleDetector()

# Create fixed containers that don't change
st.subheader("📊 Processing Statistics")

# Statistics display area (fixed position)
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    frame_metric = st.empty()
    frame_metric.metric("Frame", "0")

with stats_col2:
    people_metric = st.empty()
    people_metric.metric("People", "0")

with stats_col3:
    fps_metric = st.empty()
    fps_metric.metric("FPS", "0.0")

with stats_col4:
    time_metric = st.empty()
    time_metric.metric("Time", "0.0s")

st.markdown("---")

# Video display area
st.subheader("🎥 Live Processing Video")
video_display = st.empty()  # Single empty container for video

# Store the video display reference in session state
if st.session_state.video_placeholder is None:
    st.session_state.video_placeholder = video_display

# Progress area (only when processing)
if st.session_state.processing:
    progress_container = st.empty()
    with progress_container.container():
        st.progress(0)
        progress_text = st.empty()

# VIDEO PROCESSING
def process_video():
    if not st.session_state.processing or st.session_state.paused:
        return
    
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(VIDEO_PATH)
        if not st.session_state.cap.isOpened():
            st.error("Failed to open video file")
            st.session_state.processing = False
            return
        
        
        st.session_state.total_video_frames = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.video_fps = int(st.session_state.cap.get(cv2.CAP_PROP_FPS))
    
    if st.session_state.video_writer is None:
        width = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        st.session_state.video_writer = cv2.VideoWriter(
            OUTPUT_FILENAME, fourcc, st.session_state.video_fps, (width, height)
        )
        st.session_state.output_path = OUTPUT_FILENAME
    
    start_time = time.time()
    frames_processed = 0
    total_people = 0
    max_people = 0
    last_update_time = time.time()
    
    # Update initial display
    frame_metric.metric("Frame", "0")
    people_metric.metric("People", "0")
    fps_metric.metric("FPS", "0.0")
    time_metric.metric("Time", "0.0s")
    
    while st.session_state.processing and not st.session_state.paused:
        ret, frame = st.session_state.cap.read()
        
        if not ret:
            # End of video
            st.session_state.processing = False
            st.session_state.processing_complete = True
            
            # Calculate final statistics
            elapsed = time.time() - start_time
            avg_fps = frames_processed / elapsed if elapsed > 0 else 0
            
            st.session_state.final_stats = {
                'total_frames': frames_processed,
                'total_people': total_people,
                'max_people': max_people,
                'avg_fps': avg_fps,
                'processing_time': elapsed,
                'avg_people_per_frame': total_people / frames_processed if frames_processed > 0 else 0
            }
            
            # Update final statistics
            frame_metric.metric("Frame", frames_processed)
            people_metric.metric("People", total_people)
            fps_metric.metric("FPS", f"{avg_fps:.1f}")
            time_metric.metric("Time", f"{elapsed:.1f}s")
        
            break
        
        # Process the frame
        processed_frame, people_count, _ = st.session_state.detector.detect_people(frame)
        frames_processed += 1
        total_people += people_count
        max_people = max(max_people, people_count)
        
        # Save frame
        st.session_state.video_writer.write(processed_frame)
        st.session_state.last_frame = processed_frame
        
        # Update session state
        st.session_state.frame_count = frames_processed
        st.session_state.total_people = total_people
        st.session_state.max_people = max_people
        
        # Calculate current FPS
        current_time = time.time()
        elapsed = current_time - start_time
        current_fps = frames_processed / elapsed if elapsed > 0 else 0
        
        # Update statistics display periodically (not every frame)
        if current_time - last_update_time >= 0.5:  # Update every 0.5 seconds
            frame_metric.metric("Frame", frames_processed)
            people_metric.metric("People", total_people)
            fps_metric.metric("FPS", f"{current_fps:.1f}")
            time_metric.metric("Time", f"{elapsed:.1f}s")
            last_update_time = current_time
        
        # Update progress bar if exists
        if 'progress_container' in locals():
            progress = frames_processed / st.session_state.total_video_frames if st.session_state.total_video_frames > 0 else 0
            with progress_container.container():
                st.progress(min(progress, 1.0))
                st.text(f"Processing: {frames_processed}/{st.session_state.total_video_frames} frames ({progress*100:.1f}%)")
        
        # Update video display in the fixed container
        current_display_time = time.time()
        if current_display_time - st.session_state.last_display_time >= st.session_state.display_interval:
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            # Use the stored video placeholder
            if st.session_state.video_placeholder:
                st.session_state.video_placeholder.image(rgb_frame)
            else:
                video_display.image(rgb_frame)
            st.session_state.last_display_time = current_display_time
        
        # Control processing speed
        if st.session_state.video_fps > 0:
            time.sleep(1.0 / st.session_state.video_fps)
    
    # Cleanup
    if not st.session_state.processing:
        if st.session_state.video_writer:
            st.session_state.video_writer.release()
            st.session_state.video_writer = None
        
        if st.session_state.cap and st.session_state.cap.isOpened():
            st.session_state.cap.release()
            st.session_state.cap = None
        
        st.rerun()

# Check for auto-start
auto_start_triggered = auto_start_processing()

if auto_start_triggered:
    # Skip initial display and go directly to processing
    st.rerun()

elif not st.session_state.processing and st.session_state.frame_count == 0 and not st.session_state.processing_complete:
    if os.path.exists(VIDEO_PATH):
        st.info(f"Ready to process: **{VIDEO_PATH}**")
        
        st.markdown("Video processing will start automatically. Click **▶️ Start** in the sidebar if it doesn't begin.")
    else:
        st.error(f"Video file '{VIDEO_PATH}' not found.")

elif st.session_state.processing:
    process_video()

# ================== FINAL RESULTS SECTION ==================
elif st.session_state.processing_complete and st.session_state.final_stats:
    st.success("Processing Complete!")
    
    # Show detailed final statistics
    st.subheader("📈 Detailed Processing Summary")
    
    # Create two rows of statistics
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    
    with row1_col1:
        st.metric("Total Frames", st.session_state.final_stats['total_frames'])
        st.caption("Frames processed")
    
    with row1_col2:
        st.metric("Total People", st.session_state.final_stats['total_people'])
        st.caption("People detected")
    
    with row1_col3:
        st.metric("Processing Time", f"{st.session_state.final_stats['processing_time']:.1f}s")
        st.caption("Total time taken")
    
    with row1_col4:
        st.metric("Average FPS", f"{st.session_state.final_stats['avg_fps']:.1f}")
        st.caption("Frames per second")
    
    # Second row
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
    
    with row2_col1:
        st.metric("Max in Frame", st.session_state.final_stats['max_people'])
        st.caption("Maximum people in single frame")
    
    with row2_col2:
        st.metric("Avg per Frame", f"{st.session_state.final_stats['avg_people_per_frame']:.2f}")
        st.caption("Average people per frame")
    
    with row2_col3:
        if st.session_state.total_video_frames > 0:
            completion_rate = (st.session_state.final_stats['total_frames'] / st.session_state.total_video_frames) * 100
            st.metric("Completion", f"{completion_rate:.1f}%")
            st.caption("Video processed")
    
    with row2_col4:
        if st.session_state.final_stats['processing_time'] > 0:
            efficiency = st.session_state.final_stats['total_frames'] / st.session_state.final_stats['processing_time']
            st.metric("Efficiency", f"{efficiency:.1f} fps")
            st.caption("Processing speed")
    
    st.markdown("---")
    
    # Output Section
    output_col1, output_col2 = st.columns([2, 1])
    
    with output_col1:
        st.subheader("🎨 Detection Preview")
        if st.session_state.last_frame is not None:
            rgb_last_frame = cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_last_frame, caption="Sample frame from processed video")
    
    with output_col2:
        st.subheader("💾 Output File")
        if st.session_state.output_path and os.path.exists(st.session_state.output_path):
            file_size = os.path.getsize(st.session_state.output_path) / (1024 * 1024)
            
            st.info(f"""
            **File:** `{os.path.basename(st.session_state.output_path)}`
            **Size:** {file_size:.2f} MB
            **Duration:** {st.session_state.final_stats['processing_time']:.1f}s
            """)
            
            with open(st.session_state.output_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Video",
                    data=f.read(),
                    file_name=os.path.basename(st.session_state.output_path),
                    mime="video/mp4",
                    use_container_width=True
                )
            
            if st.button("🎬 Preview Output Video", use_container_width=True):
                st.video(st.session_state.output_path)
        else:
            st.warning("Output file not found.")
    
    # Reset button
    st.markdown("---")
    if st.button("Process Another Video", type="secondary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.rerun()

# Footer
st.markdown("---")
st.caption("**© 2025 TRAN THI THE NHAN**")