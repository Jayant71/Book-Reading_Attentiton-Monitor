from l2cs import Pipeline, render
import cv2
import numpy as np
from typing import Tuple, Optional

class GazeEstimator:
    def __init__(self, model_path: str = "L2CSNet_gaze360.pkl"):
        """
        Initialize the gaze estimator with the L2CS model.
        
        Args:
            model_path (str): Path to the L2CS model file
        """
        self.pipeline = Pipeline(
            weights=model_path,
            arch='ResNet50',
            device='cuda'  # Will fall back to CPU if CUDA is not available
        )
    
    def estimate_gaze(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate gaze direction from a frame.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (pitch, yaw) angles in degrees
        """
        try:
            # Process frame through L2CS pipeline
            results = self.pipeline.step(frame)
            
            if results is None or len(results) == 0:
                return None, None
                
            # Get the first face detection result
            result = results[0]
            
            # Extract pitch and yaw angles
            pitch = float(result.angles[0])
            yaw = float(result.angles[1])
            
            return pitch, yaw
            
        except Exception as e:
            print(f"Error in gaze estimation: {str(e)}")
            return None, None
    
    def visualize_gaze(self, frame: np.ndarray, pitch: float, yaw: float) -> np.ndarray:
        """
        Visualize gaze direction on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            pitch (float): Pitch angle in degrees
            yaw (float): Yaw angle in degrees
            
        Returns:
            np.ndarray: Frame with gaze visualization
        """
        if pitch is None or yaw is None:
            return frame
            
        return render(frame, [{'pitch': pitch, 'yaw': yaw}]) 