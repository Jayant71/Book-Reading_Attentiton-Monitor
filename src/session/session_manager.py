from datetime import datetime
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import cv2
import threading
from queue import Queue, Empty
from src.camera.camera_manager import CameraManager
from src.analysis.attention_monitor import AttentionMonitor
from src.models.gaze_estimator import GazeEstimator
from src.models.book_detector import BookDetector
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, camera_manager: CameraManager, gaze_analyzer: GazeEstimator, book_detector_model_path: str = "src/model_weights/yolo12s.pt"):
        """
        Initialize the session manager with required components.
        
        Args:
            camera_manager: Camera manager instance
            gaze_analyzer: Gaze analyzer instance
            model_path: Path to the YOLO model weights
        """
        self.camera_manager = camera_manager
        self.gaze_analyzer = gaze_analyzer
        self.book_detector = BookDetector(book_detector_model_path)
        self.attention_monitor = AttentionMonitor(book_detector_model_path)
        
        # Session state
        self.last_attention_data = None
        self.last_processed_frame = None
        self.running = False
        
        # Performance settings
        self.fps = 0
        self.frame_times = []
        self.max_frame_times = 30
        
        # Threading
        self.frame_queue = Queue(maxsize=2)
        logger.info("Session Manager initialized")

    def _process_frames(self):
        """Background thread for processing frames through gaze analysis and object detection"""
        logger.info("Starting frame processing thread")
        
        while self.running:
            try:
                frame = self._get_frame_from_queue()
                if frame is None:
                    continue
                
                # Process frame with gaze analysis
                processed_frame, gaze_results = self._analyze_gaze(frame)
                self.last_processed_frame = processed_frame
                
                # Process frame with object detection
                frame_with_detections, detections = self._detect_objects(frame)
                
                # Prepare data for attention analysis
                # attention_data = self._prepare_attention_data(gaze_results, detections)
                
                # Analyze attention
                # self.last_attention_data = self._analyze_attention(frame_with_detections, attention_data)
                
                # Log attention info periodically
                # self._log_attention_info(self.last_attention_data)
                    
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}", exc_info=True)
                self._handle_processing_error()

    def _get_frame_from_queue(self) -> Optional[np.ndarray]:
        """Get frame from queue with timeout"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except Empty:
            return None

    def _analyze_gaze(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Analyze gaze direction in the frame"""
        return self.gaze_analyzer.estimate_gaze(frame)

    def _detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Detect objects in the frame"""
        return self.book_detector.detect_objects(frame)

    def _analyze_attention(self, frame: np.ndarray, attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention based on gaze and object detection results"""
        return self.attention_monitor.analyze_attention(frame, attention_data)

    def _prepare_attention_data(self, gaze_results: Dict[str, Any], 
                              detections: List[dict]) -> Dict[str, Any]:
        """Prepare data for attention analysis"""
        attention_data = {
            'pitch': gaze_results.get('pitch'),
            'yaw': gaze_results.get('yaw'),
            'has_face': gaze_results.get('has_face', False),
            'confidence': gaze_results.get('confidence', 0.0),
            'bbox': gaze_results.get('bbox')
        }
        
        # Filter detections for books
        book_detections = self.book_detector.filter_detections(
            detections,
            min_confidence=0.5,
            target_classes=['book']
        )
        
        if book_detections:
            attention_data['has_book'] = True
            attention_data['book_detection'] = book_detections[0]
        else:
            attention_data['has_book'] = False
            
        return attention_data

    def _handle_processing_error(self):
        """Handle processing errors by creating default attention data"""
        self.last_attention_data = {
            'is_attentive': False,
            'has_face': False,
            'has_book': False,
            'message': "Error in processing"
        }

    def _update_fps(self):
        """Update FPS calculation with rolling average"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
            
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = (len(self.frame_times) - 1) / time_diff

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
                    time.sleep(0.01)
                    continue
                
                self._update_fps()

                # Queue frame for processing
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Display frame with visualization
                self._display_frame(frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Session ended by user")
                    break
                
                # Limit CPU usage
                elapsed = time.time() - last_frame_time
                if elapsed < 0.01:
                    time.sleep(0.01 - elapsed)
                last_frame_time = time.time()
                
        except KeyboardInterrupt:
            logger.info("Session interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in session: {str(e)}", exc_info=True)
        finally:
            self._cleanup(process_thread)

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display frame with attention monitoring overlay"""
        display_frame = frame.copy()
        
        if self.last_processed_frame is not None:
            display_frame = self.last_processed_frame.copy()
            
            # Add FPS counter
            cv2.putText(
                display_frame,
                f"FPS: {self.fps:.1f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            if self.last_attention_data:
                self.camera_manager.display_frame(display_frame, self.last_attention_data)
            else:
                cv2.imshow('Attention Monitor', display_frame)
        else:
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

    def _cleanup(self, process_thread: threading.Thread) -> None:
        """Clean up resources"""
        logger.info("Cleaning up resources")
        self.running = False
        
        if process_thread.is_alive():
            process_thread.join(timeout=2.0)
            
        self.camera_manager.release()
        cv2.destroyAllWindows()
