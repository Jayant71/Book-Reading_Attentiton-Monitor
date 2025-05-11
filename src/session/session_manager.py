from datetime import datetime
import time
import logging
from typing import Dict, Any
import cv2
import threading
from queue import Queue, Empty
from src.camera.camera_manager import CameraManager
from src.analysis.attention_monitor import AttentionMonitor
from gaze import GazeAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, camera_manager: CameraManager, gaze_analyzer: GazeAnalyzer, model_path: str):
        self.camera_manager = camera_manager
        self.gaze_analyzer = gaze_analyzer
        self.attention_monitor = AttentionMonitor(model_path)
        self.frame_counter = 0
        self.last_attention_data = None
        self.last_processed_frame = None
        
        # Increase processing interval to reduce CPU load (adjust as needed)
        self.PROCESS_INTERVAL = 15  # Process every 15th frame
        
        # FPS tracking
        self.fps = 0
        self.frame_times = []
        self.max_frame_times = 30  # For calculating rolling average
        
        self.running = False
        self.frame_queue = Queue(maxsize=2)  # Slightly larger queue for smoother operation
        logger.info("Session Manager initialized")

    def _process_frames(self):
        """Background thread for processing frames through L2CS and YOLO"""
        logger.info("Starting frame processing thread")
        
        while self.running:
            try:
                # Use a timeout to avoid blocking indefinitely
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Skip processing if thread has been stopped
                if not self.running:
                    break
                
                # Process with L2CS (gaze detection)
                processed_frame, gaze_results = self.gaze_analyzer.analyze_gaze(frame)
                
                # Store the processed frame
                self.last_processed_frame = processed_frame
                
                # Skip logging for better performance
                # logger.info("Processing frame with L2CS")
                
                # Prepare data for AttentionMonitor
                gaze_data_for_monitor = {
                    'pitch': gaze_results.get('pitch'),
                    'yaw': gaze_results.get('yaw'),
                    'confidence': gaze_results.get('confidence'),
                    'bbox': gaze_results.get('bbox'),
                    'has_face': gaze_results.get('has_face', True)  # Add has_face flag
                }
                
                # Analyze attention
                # logger.info("Analyzing attention status")  # Skip logging for performance
                self.last_attention_data = self.attention_monitor.analyze_attention(
                    frame,
                    gaze_data_for_monitor
                )
                
                # Only log attention changes for reduced I/O
                if self.frame_counter % 15 == 0:
                    self._log_attention_info(self.last_attention_data)
                    
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}", exc_info=True)
                # Create basic default attention data on error
                self.last_attention_data = {
                    'is_attentive': False,
                    'has_face': False,
                    'has_book': False,
                    'message': f"Error: {str(e)[:50]}"
                }

    def _update_fps(self):
        """Update FPS calculation with rolling average"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames for calculation
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
            
        # Calculate FPS from frame times
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = (len(self.frame_times) - 1) / time_diff

    def run_session(self) -> None:
        """Run the attention monitoring session"""
        try:
            logger.info("Starting camera capture session")
            self.camera_manager.start()
            
            # Start processing thread
            self.running = True
            process_thread = threading.Thread(target=self._process_frames)
            process_thread.daemon = True
            process_thread.start()
            
            last_frame_time = time.time()
            
            while True:
                frame = self.camera_manager.capture_frame()
                if frame is None:
                    time.sleep(0.01)  # Short sleep to avoid CPU spinning
                    continue
                
                # Update FPS calculation
                self._update_fps()
                self.frame_counter += 1

                # Queue frame for processing with throttling
                if self.frame_counter % self.PROCESS_INTERVAL == 0:
                    # Don't block if queue is full - just skip the frame
                    if not self.frame_queue.full():
                        # Use a smaller version of the frame for processing if needed
                        # frame_small = cv2.resize(frame, (640, 480))  # Uncomment if needed
                        self.frame_queue.put(frame)
                
                # Display the frame with visualization
                display_frame = None
                
                if self.last_processed_frame is not None:
                    # Always use the processed frame from L2CS if available
                    display_frame = self.last_processed_frame.copy()
                    
                    # Add FPS counter to the frame
                    cv2.putText(
                        display_frame,
                        f"FPS: {self.fps:.1f}",
                        (10, 120),  # Position below other text
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),  # Yellow
                        2
                    )
                    
                    if self.last_attention_data:
                        self.camera_manager.display_frame(display_frame, self.last_attention_data)
                    else:
                        cv2.imshow('Attention Monitor', display_frame)
                else:
                    # Show raw frame with FPS if no processed frame
                    display_frame = frame.copy()
                    cv2.putText(
                        display_frame,
                        f"FPS: {self.fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
                    cv2.imshow('Attention Monitor', display_frame)
                
                # Check for 'q' key to quit - use short wait time for better responsiveness
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Session ended by user")
                    break
                
                # Limit CPU usage if frames are processed very quickly
                elapsed = time.time() - last_frame_time
                if elapsed < 0.01:  # Target ~100 Hz maximum
                    time.sleep(0.01 - elapsed)
                last_frame_time = time.time()
                
        except KeyboardInterrupt:
            logger.info("Session interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in session: {str(e)}", exc_info=True)
        finally:
            logger.info("Cleaning up resources")
            self.running = False
            
            # Wait for processing thread to finish with timeout
            if process_thread.is_alive():
                process_thread.join(timeout=2.0)
                
            self.camera_manager.release()
            cv2.destroyAllWindows()

    def _log_attention_info(self, attention_data: Dict[str, Any]) -> None:
        """Log detailed attention information"""
        if not attention_data:
            return
            
        logger.info(
            "Attention Status - Message: %s, Has Face: %s, Has Book: %s, Is Attentive: %s",
            attention_data.get('message', 'Unknown'),
            attention_data.get('has_face', False),
            attention_data.get('has_book', False),
            attention_data.get('is_attentive', False)
        )
        
        gaze_dir = attention_data.get('gaze_direction')
        if gaze_dir:
            logger.info(
                "Gaze Info - Pitch: %.2f, Yaw: %.2f, Confidence: %.2f%%",
                gaze_dir.get('pitch', 0.0),
                gaze_dir.get('yaw', 0.0),
                gaze_dir.get('confidence', 0.0) * 100
            )
