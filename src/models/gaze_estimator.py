from l2cs import Pipeline, render
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import torch
import logging
import argparse
import math
import os

logger = logging.getLogger(__name__)

class GazeEstimator:
    def __init__(self, model_path: str = "src/model_weights/L2CSNet_gaze360.pkl"):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.pipeline = Pipeline(
            weights=model_path,
            arch='ResNet50',
            device=self.device
        )
    
    def estimate_gaze(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
     
        try:
            # Process frame through L2CS pipeline
            results = self.pipeline.step(frame)
            
            # Initialize default return values
            gaze_data = {
                'pitch': None,
                'yaw': None,
                'has_face': False,
                'scores': 0.0,
                'bbox': None,
                'landmarks': None
            }
            
            # Check if we have valid results
            if results is not None:
                pitch = results.pitch
                yaw = results.yaw
                bbox = results.bboxes
                landmarks = results.landmarks
                scores = results.scores
                
                # Update gaze data
                gaze_data.update({
                    'pitch': pitch,
                    'yaw': yaw,
                    'has_face': True,
                    'scores': scores,
                    'bbox': bbox,
                    'landmarks': landmarks
                })
                
                # processed_frame = self.visualize_gaze_vector(frame, pitch, yaw, bbox)
                print(gaze_data)
                processed_frame = render(frame, results)
                return processed_frame, gaze_data
            
            
            # If no valid results, return original frame
            return frame, gaze_data
        
        except ValueError as e:
            # Handle case when no faces are detected in the frame
            logger.warning("No faces detected in frame")
            return frame, {
                'pitch': None,
                'yaw': None,
                'has_face': False,
                'confidence': 0.0,
                'bbox': None,
                'landmarks': None,
                'message': 'No face detected'
            }
        
        except Exception as e:
            # Handle other unexpected errors during gaze estimation
            logger.error(f"Error in gaze estimation: {str(e)}", exc_info=True)
            return frame, {
                'pitch': None,
                'yaw': None,
                'has_face': False,
                'confidence': 0.0,
                'bbox': None,
                'landmarks': None,
                'message': f'Error: {str(e)}'
            }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Gaze Estimation Demo')
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam, URL for IP camera, or path to video/image file)')
    parser.add_argument('--model', type=str, default='src/model_weights/L2CSNet_gaze360.pkl',
                      help='Path to L2CS model weights')
    args = parser.parse_args()
    
    # Initialize estimator
    estimator = GazeEstimator(model_path=args.model)
    
    # Initialize video capture based on source type
    if args.source.isdigit():
        # Webcam input
        cap = cv2.VideoCapture(int(args.source))
    elif args.source.startswith(('http://', 'https://')):
        # IP camera input
        cap = cv2.VideoCapture(args.source)
    elif os.path.isfile(args.source):
        # Video file input
        cap = cv2.VideoCapture(args.source)
    else:
        logger.error(f"Invalid source: {args.source}")
        return

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {args.source}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # For video files, break the loop when video ends
                if os.path.isfile(args.source):
                    break
                # For live sources, continue trying to read
                continue
                
            # Process frame
            processed_frame, gaze_data = estimator.estimate_gaze(frame)
            
            # Display results
            cv2.imshow("Gaze Estimation", processed_frame)
            
            # Print gaze data if face is detected
            if gaze_data['has_face']:
                logger.info(f"Gaze Data: {gaze_data}")
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}", exc_info=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
