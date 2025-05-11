import cv2
import numpy as np
from typing import Tuple, Optional
import logging

def setup_logging(log_file: str = "attention_monitor.log") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("AttentionMonitor")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame (np.ndarray): Input frame
        target_size (Tuple[int, int]): Target (width, height)
        
    Returns:
        np.ndarray: Resized frame
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate aspect ratios
    aspect_ratio = w / h
    target_ratio = target_w / target_h
    
    if aspect_ratio > target_ratio:
        # Width is the limiting factor
        new_w = target_w
        new_h = int(new_w / aspect_ratio)
    else:
        # Height is the limiting factor
        new_h = target_h
        new_w = int(new_h * aspect_ratio)
    
    return cv2.resize(frame, (new_w, new_h))

def calculate_attention_score(gaze_angles: Tuple[Optional[float], Optional[float]], 
                            object_detections: list) -> float:
    """
    Calculate attention score based on gaze angles and object detections.
    
    Args:
        gaze_angles (Tuple[Optional[float], Optional[float]]): (pitch, yaw) angles
        object_detections (list): List of detected objects
        
    Returns:
        float: Attention score between 0 and 1
    """
    if None in gaze_angles or not object_detections:
        return 0.0
        
    pitch, yaw = gaze_angles
    
    # Normalize angles to [-1, 1] range
    pitch_norm = pitch / 90.0
    yaw_norm = yaw / 90.0
    
    # Calculate gaze score (closer to center = higher score)
    gaze_score = 1.0 - (abs(pitch_norm) + abs(yaw_norm)) / 2.0
    
    # Calculate object detection score
    detection_score = len(object_detections) / 10.0  # Normalize by expected max detections
    detection_score = min(detection_score, 1.0)
    
    # Combine scores (70% gaze, 30% detection)
    final_score = 0.7 * gaze_score + 0.3 * detection_score
    
    return max(0.0, min(1.0, final_score)) 