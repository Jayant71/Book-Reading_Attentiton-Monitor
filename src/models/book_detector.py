import cv2
import numpy as np
from typing import Tuple, List, Optional
import torch
from ultralytics import YOLO
import logging
import argparse
import os

logger = logging.getLogger(__name__)

class BookDetector:
    """A class to handle object detection using YOLO model."""
    
    def __init__(self, model_path: str = "yolov12s.pt"):
        
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def detect_objects(self, frame: np.ndarray, confidence: float = 0.5, classes: List[int] = None) -> Tuple[np.ndarray, List[dict]]:
        try:
            results = self.model.predict(frame, verbose=False, conf=confidence, classes=classes)[0]
            detections = []
            
            if results and hasattr(results, 'boxes') and len(results.boxes) > 0:
                for result in results.boxes.data.tolist():
                    try:
                        x1, y1, x2, y2, confidence, class_id = result
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(confidence),
                            'class_id': int(class_id),
                            'class_name': results.names[int(class_id)]
                        })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error processing detection result: {str(e)}")
                        continue
            else:
                logger.debug("No detections found in frame")
                print("No detections found in frame")
                
            processed_frame = self.draw_detections(frame, detections)
            return processed_frame, detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return frame, []
    
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
       
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
        
        filtered = []
        for detection in detections:
            if detection['confidence'] < min_confidence:
                continue
                
            if target_classes and detection['class_name'] not in target_classes:
                continue
                
            filtered.append(detection)
            
        return filtered 

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Book Detection Demo')
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam, URL for IP camera, or path to video/image file)')
    parser.add_argument('--model', type=str, default='src/model_weights/yolo12s.pt',
                      help='Path to YOLO model weights')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                      help='Minimum confidence threshold for detections')
    args = parser.parse_args()
    
    # Initialize detector
    detector = BookDetector(model_path=args.model)
    
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
            processed_frame, detections = detector.detect_objects(frame)
            
            
            # Display results
            cv2.imshow("Book Detection", processed_frame)
        
            
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