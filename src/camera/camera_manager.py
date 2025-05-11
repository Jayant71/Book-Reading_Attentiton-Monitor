import cv2
from typing import Optional, Tuple, Union
import numpy as np
import os

class CameraManager:
    def __init__(self, camera_source: Union[int, str] = 0):
        self.camera_source = camera_source
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_video_file = isinstance(camera_source, str) and os.path.isfile(self.camera_source)


    def start(self) -> None:
        """Start the camera capture"""
        if isinstance(self.camera_source, str): 
            if 'http' in self.camera_source:
                # For DroidCam or other IP cameras
                self.camera = cv2.VideoCapture(self.camera_source)
            elif os.path.isfile(self.camera_source):
                self.camera = cv2.VideoCapture(self.camera_source)
            else:
                raise ValueError("Invalid camera source provided")
        else:
            # For regular webcam
            self.camera = cv2.VideoCapture(self.camera_source)

        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.camera_source}")

        # Try to set camera properties for better quality (may not work with all cameras)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

    def capture_frame(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("Camera not initialized")
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def release(self) -> None:
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()

    def _draw_3d_arrow(self, img: np.ndarray, start_point: Tuple[int, int], 
                      yaw: float, pitch: float, length: int = 100) -> None:
        """Draw a 3D arrow indicating gaze direction"""
        # Convert angles to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        # Calculate 3D direction vector
        dx = length * np.sin(yaw_rad)
        dy = length * np.sin(pitch_rad)
        dz = length * np.cos(yaw_rad) * np.cos(pitch_rad)
        
        # Project end point to 2D
        end_x = int(start_point[0] + dx)
        end_y = int(start_point[1] + dy)
        
        # Calculate arrow head points (3D)
        arrow_head_length = length * 0.3
        arrow_head_width = length * 0.2
        
        # Base of the arrow head
        base_x = end_x - dx * 0.2
        base_y = end_y - dy * 0.2
        
        # Calculate perpendicular vectors for arrow head
        perp_x = -dy
        perp_y = dx
        norm = np.sqrt(perp_x*perp_x + perp_y*perp_y)
        if norm > 0:
            perp_x = perp_x/norm * arrow_head_width
            perp_y = perp_y/norm * arrow_head_width
        
        # Arrow head points
        left_x = int(base_x - perp_x)
        left_y = int(base_y - perp_y)
        right_x = int(base_x + perp_x)
        right_y = int(base_y + perp_y)
        
        # Draw main shaft with gradient color and thickness
        num_segments = 10
        for i in range(num_segments):
            t1 = i / num_segments
            t2 = (i + 1) / num_segments
            pt1_x = int(start_point[0] + dx * t1)
            pt1_y = int(start_point[1] + dy * t1)
            pt2_x = int(start_point[0] + dx * t2)
            pt2_y = int(start_point[1] + dy * t2)
            
            # Gradient from yellow to red
            color = (0, 
                    255 * (1 - t1),  # Decrease green component
                    255)  # Full red
            
            # Varying thickness
            thickness = int(5 * (1 - t1) + 2)
            cv2.line(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, thickness)
        
        # Draw arrow head
        cv2.fillPoly(img, [np.array([
            [end_x, end_y],
            [left_x, left_y],
            [right_x, right_y]
        ])], (0, 0, 255))

    def display_frame(self, frame: np.ndarray, attention_data: dict) -> None:
        """Display frame with attention monitoring overlay"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Create a larger canvas to accommodate the info panel above the video
        info_panel_height = 140
        total_height = h + info_panel_height
        canvas = np.zeros((total_height, w, 3), dtype=np.uint8)
        
        # Place the video frame below the info panel
        canvas[info_panel_height:, :] = display_frame
        
        # Draw face bounding box and gaze direction if face is detected
        if attention_data['has_face'] and attention_data.get('bounding_box'):
            bbox = attention_data['bounding_box']
            
            # Convert relative coordinates to absolute pixel coordinates
            x = int(bbox['Left'] * w)
            y = int(bbox['Top'] * h) + info_panel_height  # Adjust y coordinate for info panel
            width = int(bbox['Width'] * w)
            height = int(bbox['Height'] * h)
            
            # Draw face bounding box
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 255, 0), 3)
            
            # Draw 3D gaze direction arrow
            if attention_data.get('gaze_direction'):
                # Center point of the face
                center_x = x + width // 2
                center_y = y + height // 2
                
                gaze_dir = attention_data['gaze_direction']
                
                # Draw 3D arrow with length proportional to face size
                arrow_length = int(max(width, height) * 1.5)
                self._draw_3d_arrow(
                    canvas,
                    (center_x, center_y),
                    gaze_dir['yaw'],
                    gaze_dir['pitch'],
                    arrow_length
                )
        
        # Draw book bounding box if detected
        if attention_data.get('book_box'):
            book_box = attention_data['book_box']
            x1 = int(book_box['x1'] * w)
            y1 = int(book_box['y1'] * h) + info_panel_height  # Adjust y coordinate for info panel
            x2 = int(book_box['x2'] * w)
            y2 = int(book_box['y2'] * h) + info_panel_height  # Adjust y coordinate for info panel
            
            # Draw book bounding box in blue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # Add book state text
            book_state = attention_data.get('book_state', 'unknown')
            state_color = (0, 255, 0) if book_state == "opened" else (0, 0, 255)
            cv2.putText(canvas, f"Book: {book_state}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
        
        # Create info panel background
        cv2.rectangle(canvas, (0, 0), (w, info_panel_height), (0, 0, 0), -1)
        
        # Display attention status
        status_color = (0, 255, 0) if attention_data.get('is_attentive', False) else (0, 0, 255)
        cv2.putText(
            canvas,
            f"Status: {attention_data.get('message', 'Unknown')}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2
        )
        
        # Add face detection status
        face_status = "Face Detected" if attention_data['has_face'] else "No Face Detected"
        cv2.putText(
            canvas,
            face_status,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Add book detection status
        book_status = "Book Detected" if attention_data.get('has_book', False) else "No Book Detected"
        cv2.putText(
            canvas,
            book_status,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Add gaze direction values if available
        if attention_data.get('gaze_direction'):
            gaze_dir = attention_data['gaze_direction']
            cv2.putText(
                canvas,
                f"Yaw: {gaze_dir['yaw']:.1f}°, Pitch: {gaze_dir['pitch']:.1f}°",
                (w - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        # Create window with properties
        cv2.namedWindow('Attention Monitor', cv2.WINDOW_NORMAL)
        
        # Calculate scaling factor to fit the window
        target_width = 640
        target_height = 480
        
        # Calculate scaling factors
        scale_x = target_width / w
        scale_y = target_height / total_height
        
        # Use the smaller scaling factor to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(w * scale)
        new_height = int(total_height * scale)
        
        # Resize the canvas
        resized_canvas = cv2.resize(canvas, (new_width, new_height))
        
        # Create a black background
        background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the resized canvas
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Place the resized canvas in the center of the background
        background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_canvas
        
        # Display the frame
        cv2.imshow('Attention Monitor', background)
