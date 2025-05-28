# src/session/session_manager.py

import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import cv2
import threading
from queue import Queue, Empty
import numpy as np

from src.camera.camera_manager import CameraManager
from src.analysis.attention_monitor import AttentionMonitor
from src.models.gaze_estimator import GazeEstimator
from src.models.book_detector import BookDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, camera_manager: CameraManager, gaze_analyzer: GazeEstimator, book_detector_model_path: str):
        """
        Initialize the session manager with required components.
        The attention_monitor no longer needs the model path.
        """
        self.camera_manager = camera_manager
        self.gaze_analyzer = gaze_analyzer
        self.book_detector = BookDetector(book_detector_model_path)
        self.attention_monitor = AttentionMonitor()  # No longer needs model_path
        
        self.last_attention_data = {}
        self.last_gaze_frame = None
        self.last_book_frame = None
        self.running = False
        
        self.fps = 0
        self.frame_times = []
        self.max_frame_times = 30
        
        self.frame_queue = Queue(maxsize=2)
        logger.info("Session Manager initialized with corrected architecture")

    def _process_frames(self):
        """Background thread for processing frames in a clean pipeline."""
        logger.info("Starting frame processing thread")
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:
                    continue

                # 1. Gaze Analysis
                gaze_frame, gaze_data = self.gaze_analyzer.estimate_gaze(frame.copy())
                self.last_gaze_frame = gaze_frame

                # 2. Book Detection
                book_frame, book_detections = self.book_detector.detect_objects(frame.copy())
                self.last_book_frame = book_frame

                # 3. Attention Analysis (now receives data, not the frame)
                self.last_attention_data = self.attention_monitor.analyze_attention(
                    gaze_data, 
                    book_detections,
                    frame.shape
                )
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}", exc_info=True)
                self.last_attention_data = {'message': "Error in processing"}

    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
            
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            self.fps = (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0

    def run_session(self) -> None:
        """Run the attention monitoring session."""
        try:
            logger.info("Starting camera capture session")
            self.camera_manager.start()
            
            self.running = True
            process_thread = threading.Thread(target=self._process_frames)
            process_thread.daemon = True
            process_thread.start()
            
            while self.running:
                frame = self.camera_manager.capture_frame()
                if frame is None:
                    if self.camera_manager.is_video_file:
                        break # End of video
                    time.sleep(0.01)
                    continue
                
                self._update_fps()

                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                self._display_frame(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Session ended by user")
                    break
                
        except (KeyboardInterrupt, StopIteration):
            logger.info("Session interrupted")
        except Exception as e:
            logger.error(f"Unexpected error in session: {str(e)}", exc_info=True)
        finally:
            self._cleanup(process_thread)

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display frame with combined overlays."""
        display_frame = self.last_gaze_frame if self.last_gaze_frame is not None else frame.copy()

        # Overlay book detections
        if 'book_box' in self.last_attention_data and self.last_attention_data['book_box']:
            x1, y1, x2, y2 = self.last_attention_data['book_box']
            state = self.last_attention_data.get('book_state', 'book')
            color = (0, 255, 0) if self.last_attention_data.get('is_attentive') else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, state, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display FPS
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display Status Message
        status_message = self.last_attention_data.get('message', 'Initializing...')
        if 'is_attentive' in self.last_attention_data and self.last_attention_data['book_state'] == 'opened':
             if self.last_attention_data['is_attentive']:
                 status_message = "Attentive"
                 status_color = (0, 255, 0) # Green
             else:
                 status_message = "Distracted"
                 status_color = (0, 0, 255) # Red
        else:
            status_color = (0, 255, 255) # Yellow for other states
            
        cv2.putText(display_frame, f"Status: {status_message}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow('Attention Monitor', display_frame)

    def _cleanup(self, process_thread: threading.Thread) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources")
        self.running = False
        if process_thread.is_alive():
            process_thread.join(timeout=1.0)
        self.camera_manager.release()