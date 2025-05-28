from typing import Dict, Any, Optional, List
import numpy as np
import cv2

class AttentionAnalyzer:
    # Gaze direction thresholds
    CONFIDENCE_THRESHOLD = 90.0
    
    def __init__(self):
        pass

    def analyze_attention(self, response: Dict[str, Any]) -> Dict[str, Any]:
        attention_data = {
            'is_looking': False,
            'gaze_direction': 'Unknown',
            'confidence': 0.0,
            'bounding_box': None,
            'eye_direction': None
        }

        face_details = response.get('FaceDetails', [])
        
        # Check if face is detected
        if not face_details:
            attention_data['gaze_direction'] = 'No face detected'
            return attention_data

        face = face_details[0]
        eye_direction = face.get('EyeDirection')
        bounding_box = face.get('BoundingBox')
        
        if not eye_direction or not bounding_box:
            attention_data['gaze_direction'] = 'Eye direction not detected'
            return attention_data
        
        confidence = eye_direction.get('Confidence', 0.0)
        if confidence < self.CONFIDENCE_THRESHOLD:
            attention_data['gaze_direction'] = 'Low confidence in eye direction'
            return attention_data
        
        yaw = float(eye_direction.get('Yaw', 0.0))  # Removed inversion
        pitch = float(eye_direction.get('Pitch', 0.0))
        
        # Determine gaze direction
        gaze_direction = self._get_gaze_direction(yaw, pitch)
        
        # Update attention data with gaze information
        attention_data.update({
            'is_looking': True,
            'gaze_direction': gaze_direction,
            'confidence': confidence,
            'bounding_box': bounding_box,
            'eye_direction': {'yaw': yaw, 'pitch': pitch}
        })
        
        return attention_data
    
    def _get_gaze_direction(self, yaw: float, pitch: float) -> str:
        """Determine gaze direction based on yaw and pitch values"""
        if abs(yaw) < 5 and abs(pitch) < 5:
            return "Center"
        direction = []
        if yaw < -5: direction.append("Left")
        elif yaw > 5: direction.append("Right")
        if pitch < -5: direction.append("Up")
        elif pitch > 5: direction.append("Down")
        return " ".join(direction) if direction else "Center"
