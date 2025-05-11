import numpy as np
from ultralytics import YOLO
from typing import Dict, Any, Tuple, Optional, List
import logging
import math
import time

# Configure logging
logger = logging.getLogger(__name__)

class AttentionMonitor:
    def __init__(self, model_path: str = "src/model_weights/yolo12s.pt"):
        # Load YOLO with optimization flags
        self.yolo_model = YOLO(model_path)
        
        # Set optimized inference parameters
        self.yolo_conf_threshold = 0.25  # Lower confidence threshold for faster inference
        self.yolo_iou_threshold = 0.7  # IOU threshold
        
        # Pre-calculate some constants for 3D calculations
        self.z_near = 1.0
        self.z_far = 100.0
        self.epsilon = 1e-6  # Small value to avoid division by zero
        
        # Performance tracking
        self._last_process_time = 0
        self._process_times = []
        self._max_process_times = 30  # Keep last 30 processing times
        
        logger.info(f"Initialized custom YOLO model from {model_path}")
        logger.info(f"YOLO confidence threshold: {self.yolo_conf_threshold}")
        logger.info(f"YOLO IOU threshold: {self.yolo_iou_threshold}")

    def _calculate_3d_gaze_vector(self, pitch: float, yaw: float) -> np.ndarray:
        """
        Calculate 3D gaze vector from pitch and yaw angles.
        
        Args:
            pitch: Pitch angle in degrees (up/down)
            yaw: Yaw angle in degrees (left/right)
            
        Returns:
            3D unit vector [x, y, z] representing gaze direction
        """
        try:
            # Convert angles to radians
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            
            # Use optimized calculation (avoid redundant computations)
            cos_pitch = np.cos(pitch_rad)
            sin_pitch = np.sin(pitch_rad)
            cos_yaw = np.cos(yaw_rad)
            sin_yaw = np.sin(yaw_rad)
            
            # Calculate 3D direction vector
            # Standard convention: x=right, y=down, z=forward
            x = cos_pitch * sin_yaw
            y = sin_pitch
            z = cos_pitch * cos_yaw
            
            # Create pre-normalized vector (avoids extra array creation)
            vector = np.array([x, y, z], dtype=np.float32)
            norm = np.sqrt(x*x + y*y + z*z)  # Optimized norm calculation
            
            if norm > self.epsilon:
                vector /= norm
                
            logger.debug(f"Calculated 3D gaze vector: {vector}")
            return vector
            
        except Exception as e:
            logger.error(f"Error calculating 3D gaze vector: {str(e)}")
            return np.array([0, 0, 1], dtype=np.float32)  # Default forward direction

    def _calculate_gaze_line(self, face_center: Tuple[float, float], 
                           pitch: float, yaw: float, frame_shape: Tuple[int, int]) -> Tuple[Tuple, np.ndarray]:
        """
        Calculate start and end points for 3D gaze line visualization.
        
        Args:
            face_center: (x, y) in image coordinates
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees
            frame_shape: (height, width) of the frame
            
        Returns:
            Tuple containing:
            - start_point: (x, y) face center in image space
            - 3D gaze vector normalized
            - end_point: (x, y) for visualization in image space
        """
        try:
            # Get 3D gaze vector
            gaze_vector = self._calculate_3d_gaze_vector(pitch, yaw)
            
            # For visualization: project back to 2D
            length = min(frame_shape[0], frame_shape[1]) * 0.3  # Use smaller dimension for better scaling
            
            # Optimize calculation
            dx = length * gaze_vector[0]
            dy = length * gaze_vector[1]
            
            # Calculate end point for visualization (avoid unnecessary conversions)
            start_point = np.array(face_center, dtype=np.float32)
            end_point = np.array([start_point[0] + dx, start_point[1] + dy], dtype=np.int32)
            
            logger.debug(f"Gaze line - Start: {start_point}, End: {end_point}, Vector: {gaze_vector}")
            return start_point, gaze_vector, end_point
            
        except Exception as e:
            logger.error(f"Error calculating gaze line: {str(e)}")
            return (0, 0), np.array([0, 0, 1]), (0, 0)

    def _ray_box_intersection(self, ray_origin: np.ndarray, ray_direction: np.ndarray, 
                            box_coords: Dict[str, float], frame_shape: Tuple[int, int]) -> bool:
        """
        Check if a 3D ray intersects with a 2D bounding box extended to 3D.
        Optimized version with early returns and reduced computations.
        """
        try:
            h, w = frame_shape[:2]
            
            # Denormalize box coordinates to pixel values
            x1 = box_coords['x1'] * w
            y1 = box_coords['y1'] * h
            x2 = box_coords['x2'] * w
            y2 = box_coords['y2'] * h
            
            logger.debug(f"Checking intersection with box: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Convert ray_origin to 3D
            ray_origin_3d = np.array([ray_origin[0], ray_origin[1], 0], dtype=np.float32) if len(ray_origin) == 2 else ray_origin
            
            # Check X planes (left/right)
            if abs(ray_direction[0]) > self.epsilon:
                t_x1 = (x1 - ray_origin_3d[0]) / ray_direction[0]
                if t_x1 > 0:
                    intersect = ray_origin_3d + t_x1 * ray_direction
                    if (y1 <= intersect[1] <= y2 and self.z_near <= intersect[2] <= self.z_far):
                        logger.debug("Intersection found on left X plane")
                        return True
                        
                t_x2 = (x2 - ray_origin_3d[0]) / ray_direction[0]
                if t_x2 > 0:
                    intersect = ray_origin_3d + t_x2 * ray_direction
                    if (y1 <= intersect[1] <= y2 and self.z_near <= intersect[2] <= self.z_far):
                        logger.debug("Intersection found on right X plane")
                        return True
            
            # Check Y planes (top/bottom)
            if abs(ray_direction[1]) > self.epsilon:
                t_y1 = (y1 - ray_origin_3d[1]) / ray_direction[1]
                if t_y1 > 0:
                    intersect = ray_origin_3d + t_y1 * ray_direction
                    if (x1 <= intersect[0] <= x2 and self.z_near <= intersect[2] <= self.z_far):
                        logger.debug("Intersection found on top Y plane")
                        return True
                        
                t_y2 = (y2 - ray_origin_3d[1]) / ray_direction[1]
                if t_y2 > 0:
                    intersect = ray_origin_3d + t_y2 * ray_direction
                    if (x1 <= intersect[0] <= x2 and self.z_near <= intersect[2] <= self.z_far):
                        logger.debug("Intersection found on bottom Y plane")
                        return True
            
            # Check Z planes (near/far)
            if abs(ray_direction[2]) > self.epsilon:
                t_z1 = (self.z_near - ray_origin_3d[2]) / ray_direction[2]
                if t_z1 > 0:
                    intersect = ray_origin_3d + t_z1 * ray_direction
                    if (x1 <= intersect[0] <= x2 and y1 <= intersect[1] <= y2):
                        logger.debug("Intersection found on near Z plane")
                        return True
                        
                t_z2 = (self.z_far - ray_origin_3d[2]) / ray_direction[2]
                if t_z2 > 0:
                    intersect = ray_origin_3d + t_z2 * ray_direction
                    if (x1 <= intersect[0] <= x2 and y1 <= intersect[1] <= y2):
                        logger.debug("Intersection found on far Z plane")
                        return True
            
            logger.debug("No intersection found")
            return False
            
        except Exception as e:
            logger.error(f"Error in ray-box intersection: {str(e)}")
            return False

    def analyze_attention(self, frame: np.ndarray, 
                         gaze_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if person is looking at an opened book in the frame.
        Returns attention status and relevant data.
        """
        start_time = time.time()
        
        try:
            h, w = frame.shape[:2]
            
            # Initialize with basic info
            attention_status = {
                'is_attentive': False,
                'has_face': False,
                'has_book': False,
                'book_state': None,
                'gaze_direction': None,
                'face_box': None,
                'book_box': None,
                'message': "No face detected"
            }

            # Check if the gaze analyzer detected a face
            if not gaze_data.get('has_face', True):
                logger.debug("No face detected in gaze data")
                return attention_status

            # Quick returns for missing data
            pitch = gaze_data.get('pitch')
            yaw = gaze_data.get('yaw')
            if pitch is None or yaw is None:
                logger.debug("Missing pitch or yaw data")
                return attention_status
                
            face_bbox_coords = gaze_data.get('bbox')
            attention_status['has_face'] = True
            attention_status['face_box'] = face_bbox_coords
            
            logger.debug(f"Face detected - Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
            
            # Calculate face center
            if face_bbox_coords:
                x1, y1, x2, y2 = face_bbox_coords
                face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                logger.debug(f"Face center: {face_center}")
            else:
                face_center = (w / 2, h / 2)
                attention_status['message'] = "Face detected, but bbox missing. Using frame center."
                logger.warning("Using frame center as face center (bbox missing)")

            # Calculate 3D gaze
            start_point, gaze_vector, end_point = self._calculate_gaze_line(
                face_center, pitch, yaw, frame.shape
            )

            # Run YOLO with optimized settings
            logger.debug("Running YOLO detection")
            yolo_results = self.yolo_model(
                frame, 
                verbose=False,
                conf=self.yolo_conf_threshold,
                iou=self.yolo_iou_threshold,
            )
            
            # Quick exit if no books detected
            if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
                attention_status['has_book'] = False
                attention_status['message'] = "No book detected"
                logger.debug("No books detected in frame")
                attention_status['gaze_direction'] = {
                    'pitch': pitch,
                    'yaw': yaw,
                    'confidence': gaze_data.get('confidence', 0.0),
                    'start_point': start_point.tolist() if isinstance(start_point, np.ndarray) else list(start_point),
                    'end_point': end_point.tolist() if isinstance(end_point, np.ndarray) else list(end_point),
                    'vector': gaze_vector.tolist()
                }
                return attention_status
                
            # Book found
            attention_status['has_book'] = True
            logger.debug(f"Detected {len(yolo_results[0].boxes)} objects")
            
            # Find opened books efficiently
            opened_book_found = False
            book_box_data = None
            
            # Process all detected objects
            boxes = yolo_results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                book_class = int(box.cls.item())
                confidence = float(box.conf.item())
                
                logger.debug(f"Object {i}: Class {book_class}, Confidence: {confidence:.2f}")
                
                # If opened book (class 0), prioritize it
                if book_class == 0:
                    opened_book_found = True
                    attention_status['book_state'] = "opened"
                    coords = box.xyxyn[0].tolist()
                    book_box_data = {
                        'x1': coords[0],
                        'y1': coords[1],
                        'x2': coords[2],
                        'y2': coords[3]
                    }
                    attention_status['book_box'] = book_box_data
                    logger.debug(f"Found opened book at {book_box_data}")
                    break
            
            # If no opened book found, use the first book (likely closed)
            if not opened_book_found and len(boxes) > 0:
                first_box = boxes[0]
                attention_status['book_state'] = "closed"
                coords = first_box.xyxyn[0].tolist()
                book_box_data = {
                    'x1': coords[0],
                    'y1': coords[1],
                    'x2': coords[2],
                    'y2': coords[3]
                }
                attention_status['book_box'] = book_box_data
                attention_status['message'] = "Closed book detected"
                logger.debug(f"Using closed book at {book_box_data}")
                
            # Check intersection if opened book found
            if attention_status['book_state'] == "opened" and book_box_data:
                logger.debug("Checking gaze intersection with opened book")
                is_intersecting = self._ray_box_intersection(
                    start_point,
                    gaze_vector,
                    book_box_data,
                    frame.shape
                )
                
                # Update attention status based on intersection
                if is_intersecting:
                    attention_status['is_attentive'] = True
                    attention_status['message'] = "Attentive (looking at open book)"
                    logger.info("User is attentive - looking at open book")
                else:
                    attention_status['is_attentive'] = False
                    attention_status['message'] = "Distracted (open book detected)"
                    logger.info("User is distracted - not looking at open book")
                
            # Store gaze info
            attention_status['gaze_direction'] = {
                'pitch': pitch,
                'yaw': yaw,
                'confidence': gaze_data.get('confidence', 0.0),
                'start_point': start_point.tolist() if isinstance(start_point, np.ndarray) else list(start_point),
                'end_point': end_point.tolist() if isinstance(end_point, np.ndarray) else list(end_point),
                'vector': gaze_vector.tolist()
            }
            
            # Track processing time
            process_time = time.time() - start_time
            self._process_times.append(process_time)
            if len(self._process_times) > self._max_process_times:
                self._process_times.pop(0)
            
            avg_process_time = sum(self._process_times) / len(self._process_times)
            logger.debug(f"Frame processed in {process_time*1000:.1f}ms (avg: {avg_process_time*1000:.1f}ms)")
            
            return attention_status
            
        except Exception as e:
            logger.error(f"Error in attention analysis: {str(e)}", exc_info=True)
            return attention_status 