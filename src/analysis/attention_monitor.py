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

    def _calculate_gaze_line(self, face_center: Tuple[float, float], 
                           pitch: float, yaw: float, 
                           face_bbox: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Calculate start and end points for 2D gaze line visualization.
        
        NOTE: This version corrects for an observed axis swap from the underlying gaze model.
        It maps the model's 'pitch' to the horizontal axis (dx) and the model's 'yaw'
        to the vertical axis (dy) to produce correct, intuitive visualization.
        """
        start_point = np.array(face_center, dtype=np.int32)
        
        # Determine the length of the gaze line using the width of the face bounding box, scaled for visibility.
        x1, _, x2, _ = face_bbox
        length = float(x2 - x1) * 2.0

        # --- DEFINITIVE FIX FOR AXIS SWAP ---
        # Based on consistent user feedback, the model's axes are swapped.
        
        # Horizontal movement (dx) is controlled by the model's 'pitch'.
        dx = -length * np.sin(pitch) * np.cos(yaw)
        
        # Vertical movement (dy) is controlled by the model's 'yaw'.
        dy = -length * np.sin(yaw)
        
        # Calculate the final end point.
        # We add dy because the screen's y-axis increases downwards.
        end_point = np.array([start_point[0] + dx, start_point[1] + dy], dtype=np.int32)
        
        return tuple(start_point.tolist()), tuple(end_point.tolist())

    def _is_gaze_endpoint_in_box(self, gaze_endpoint: Tuple[int, int], 
                                   box_coords: Tuple[int, int, int, int]) -> bool:
        """
        Check if the gaze endpoint is inside the given bounding box.
        """
        if gaze_endpoint is None or box_coords is None:
            return False

        px, py = gaze_endpoint
        x1, y1, x2, y2 = box_coords
        
        isAttentive = (x1 <= px <= x2) and (y1 <= py <= y2)
        
        return isAttentive

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

        pitch, yaw, face_bbox_list = gaze_data.get('pitch'), gaze_data.get('yaw'), gaze_data.get('bbox')
        
        has_face_bbox = face_bbox_list is not None and face_bbox_list.size > 0
        
        attention_status.update({'has_face': True, 'face_box': face_bbox_list[0] if has_face_bbox else None})

        if pitch is None or yaw is None or not has_face_bbox:
            attention_status['message'] = "Gaze estimation failed"
            return attention_status
        
        # Use the first detected face for analysis
        face_bbox_coords = face_bbox_list[0]
        x1, y1, x2, y2 = face_bbox_coords 
        face_center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Using pitch and yaw from the first detected face.
        current_pitch = pitch[0]
        current_yaw = yaw[0]

        start_point, end_point = self._calculate_gaze_line(face_center, current_pitch, current_yaw, face_bbox_coords)
        attention_status['gaze_direction'] = {'pitch': current_pitch, 'yaw': current_yaw, 'start_point': start_point, 'end_point': end_point}
        
        opened_book_detections = [d for d in book_detections if d.get('class_name') == 'opened']
        
        if opened_book_detections:
            attention_status.update({'has_book': True, 'book_state': 'opened'})
            book_box = opened_book_detections[0]['bbox']
            attention_status['book_box'] = book_box

            is_attentive = self._is_gaze_endpoint_in_box(end_point, book_box)
            
            if is_attentive:
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