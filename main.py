import os
import sys
import logging
from pathlib import Path

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
from gaze import GazeAnalyzer

def main():
    try:
        logger.info("Starting Book Attention Monitoring System")
        
        # Initialize components
        logger.info("Initializing camera manager")
        # source = "http://192.168.29.181:4747/video"
        source = 0
        # source = "sample.mp4"

        camera_manager = CameraManager(source)
        
        logger.info("Initializing L2CS gaze analyzer")
        gaze_analyzer = GazeAnalyzer(weights_path='L2CSNet_gaze360.pkl')
        
        logger.info("Initializing session manager")
        model_path = "best.onnx"
        session_manager = SessionManager(camera_manager, gaze_analyzer, model_path=model_path)
        
        # Run the session
        logger.info("Starting attention monitoring session")
        session_manager.run_session()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
