# src/analysis/attention_monitor.py

import numpy as np
from typing import Dict, Any, Tuple, List
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

class AttentionMonitor:
    def __init__(self):
        """
        Initializes the AttentionMonitor.
        This class is now decoupled from model loading and focuses solely on analysis.
        """
        self.epsilon = 1e-6  # Small value to avoid division by zero
        
        # Performance tracking
        self._process_times = []
        self._max_process_times = 30  # Keep last 30 processing times
        
        logger.info("Initialized Attention Monitor.")

    def _calculate_3d_gaze_vector(self, pitch: float, yaw: float) -> np.ndarray:
        """
        Calculate 3D gaze vector from pitch and yaw angles.
        Pitch > 0 is Down, Yaw > 0 is Right.
        """
        try:
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            
            cos_pitch = np.cos(pitch_rad)
            sin_pitch = np.sin(pitch_rad)
            cos_yaw = np.cos(yaw_rad)
            sin_yaw = np.sin(yaw_rad)
            
            x = cos_pitch * sin_yaw
            y = sin_pitch  # Positive y is down, which matches screen coordinates
            z = cos_pitch * cos_yaw
            
            vector = np.array([x, y, z], dtype=np.float32)
            norm = np.linalg.norm(vector)
            
            if norm > self.epsilon:
                vector /= norm
                
            return vector
            
        except Exception as e:
            logger.error(f"Error calculating 3D gaze vector: {str(e)}")
            return np.array([0, 0, 1], dtype=np.float32)

    def _calculate_gaze_line(self, face_center: Tuple[float, float], 
                           pitch: float, yaw: float, frame_shape: Tuple[int, int, int]) -> Tuple[Tuple, np.ndarray, Tuple]:
        """
        Calculate start and end points for 2D gaze line visualization.
        """
        gaze_vector = self._calculate_3d_gaze_vector(pitch, yaw)
        length = min(frame_shape[0], frame_shape[1]) * 0.4 # Increased length for better visualization
        
        # dx and dy are based on the 2D projection of the gaze vector
        dx = length * gaze_vector[0] 
        dy = length * gaze_vector[1]
        
        start_point = np.array(face_center, dtype=np.float32)
        # FIX: The Y-axis for screen coordinates increases downwards. No inversion needed.
        end_point = np.array([start_point[0] + dx, start_point[1] + dy], dtype=np.int32)
        
        return start_point, gaze_vector, end_point

    def _gaze_intersects_box(self, gaze_origin: Tuple[float, float], gaze_vector: np.ndarray,
                             box_coords: Tuple[int, int, int, int]) -> bool:
        """
        Check if a 2D gaze ray intersects with a 2D bounding box.
        This is the new, corrected intersection logic.
        """
        x1, y1, x2, y2 = box_coords
        ox, oy = gaze_origin

        # The 2D direction vector on the screen
        dx = gaze_vector[0]
        dy = gaze_vector[1]

        # If origin of gaze is already inside the book, count it as intersection.
        if x1 <= ox <= x2 and y1 <= oy <= y2:
            return True

        # Check for intersection with each of the four sides of the bounding box
        # We are checking if the ray starting from gaze_origin intersects the box edges
        t_values = []
        # Top edge
        if abs(dy) > self.epsilon:
            t = (y1 - oy) / dy
            if t > 0 and x1 <= (ox + t * dx) <= x2: t_values.append(t)
        # Bottom edge
        if abs(dy) > self.epsilon:
            t = (y2 - oy) / dy
            if t > 0 and x1 <= (ox + t * dx) <= x2: t_values.append(t)
        # Left edge
        if abs(dx) > self.epsilon:
            t = (x1 - ox) / dx
            if t > 0 and y1 <= (oy + t * dy) <= y2: t_values.append(t)
        # Right edge
        if abs(dx) > self.epsilon:
            t = (x2 - ox) / dx
            if t > 0 and y1 <= (oy + t * dy) <= y2: t_values.append(t)

        return len(t_values) > 0

    def analyze_attention(self, gaze_data: Dict[str, Any], 
                          book_detections: List[Dict[str, Any]],
                          frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Analyze if person is looking at an opened book.
        """
        start_time = time.time()
        
        h, w, _ = frame_shape
        
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

        if not gaze_data.get('has_face'):
            return attention_status

        pitch, yaw, face_bbox_coords = gaze_data.get('pitch'), gaze_data.get('yaw'), gaze_data.get('bbox')
        attention_status.update({'has_face': True, 'face_box': face_bbox_coords})

        if pitch is None or yaw is None:
            attention_status['message'] = "Gaze estimation failed"
            return attention_status

        face_center = (w / 2, h / 2)
        if face_bbox_coords is not None and len(face_bbox_coords) > 0:
            x1, y1, x2, y2 = face_bbox_coords[0] 
            face_center = ((x1 + x2) / 2, (y1 + y2) / 2)

        current_pitch = pitch[0] if pitch is not None and len(pitch) > 0 else 0.0
        current_yaw = yaw[0] if yaw is not None and len(yaw) > 0 else 0.0

        start_point, gaze_vector, end_point = self._calculate_gaze_line(face_center, current_pitch, current_yaw, frame_shape)
        attention_status['gaze_direction'] = {'pitch': current_pitch, 'yaw': current_yaw, 'start_point': start_point.tolist(), 'end_point': end_point.tolist()}
        
        # FIX: Correctly check for 'opened_book' instead of 'closed_book'
        opened_book_detections = [d for d in book_detections if d.get('class_name') == 'opened_book']
        
        if opened_book_detections:
            attention_status.update({'has_book': True, 'book_state': 'opened'})
            book_box = opened_book_detections[0]['bbox']
            attention_status['book_box'] = book_box

            is_intersecting = self._gaze_intersects_box(face_center, gaze_vector, book_box)
            
            if is_intersecting:
                attention_status.update({'is_attentive': True, 'message': 'Attentive'})
            else:
                attention_status.update({'is_attentive': False, 'message': 'Distracted'})
        elif book_detections:
            state = book_detections[0].get('class_name', 'book')
            attention_status.update({'has_book': True, 'book_state': state, 'book_box': book_detections[0]['bbox'], 'message': f'{state.replace("_", " ").title()} Detected'})
        else:
            attention_status.update({'has_book': False, 'message': 'No book detected'})

        process_time = time.time() - start_time
        self._process_times.append(process_time)
        if len(self._process_times) > self._max_process_times:
            self._process_times.pop(0)
        
        return attention_status