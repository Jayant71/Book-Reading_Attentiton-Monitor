import os
import sys
import logging
from pathlib import Path

YOLO_MODEL_PATH = "src/model_weights/yolo12s.pt"
L2CS_MODEL_PATH = "src/model_weights/L2CSNet_gaze360.pkl"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attention_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to PYTHONPATH
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

from src.camera.camera_manager import CameraManager
from src.session.session_manager import SessionManager
from src.models.gaze_estimator import GazeEstimator

def main():
    try:
        logger.info("Starting Book Attention Monitoring System")
        
        # Initialize components
        logger.info("Initializing camera manager")
        source = 0  # Default webcam
        # source = "sample.mp4"
        # source = "http://192.168.29.127:4747/video"
        camera_manager = CameraManager(source)
        
        logger.info("Initializing gaze estimator")
        gaze_estimator = GazeEstimator(
            model_path=L2CS_MODEL_PATH
        )
        
        logger.info("Initializing session manager")
        
        session_manager = SessionManager(
            camera_manager=camera_manager,
            gaze_analyzer=gaze_estimator,
            book_detector_model_path=YOLO_MODEL_PATH
        )
        
        # Run the session
        logger.info("Starting attention monitoring session")
        session_manager.run_session()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    main()
