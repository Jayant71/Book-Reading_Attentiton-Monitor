import cv2
import numpy as np
from typing import Tuple, List, Optional
import torch
from ultralytics import YOLO

class ObjectDetector:
    """A class to handle object detection using YOLO model."""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the object detector.
        
        Args:
            model_path (str): Path to the YOLO model weights
        """
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect objects in the given frame.
        
        Args:
            frame (np.ndarray): Input frame to detect objects in
            
        Returns:
            Tuple[np.ndarray, List[dict]]: Processed frame with detections and list of detection results
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': results.names[int(class_id)]
            })
            
        return frame, detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            detections (List[dict]): List of detection results
            
        Returns:
            np.ndarray: Frame with drawn detections
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame
    
    def filter_detections(self, detections: List[dict], 
                         min_confidence: float = 0.5,
                         target_classes: Optional[List[str]] = None) -> List[dict]:
        """
        Filter detections based on confidence and target classes.
        
        Args:
            detections (List[dict]): List of detection results
            min_confidence (float): Minimum confidence threshold
            target_classes (Optional[List[str]]): List of target class names to keep
            
        Returns:
            List[dict]: Filtered detection results
        """
        filtered = []
        for detection in detections:
            if detection['confidence'] < min_confidence:
                continue
                
            if target_classes and detection['class_name'] not in target_classes:
                continue
                
            filtered.append(detection)
            
        return filtered 