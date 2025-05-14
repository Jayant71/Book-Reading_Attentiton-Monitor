# Chapter 2: Session Manager

Welcome back! In [Chapter 1: Application Entry Point](01_application_entry_point_.md), we saw how the `main.py` script is the starting point for our project. It's like flipping the main switch. But `main.py` doesn't actually *do* the continuous work of monitoring attention. Instead, it sets up the necessary tools and then hands off control to the component that manages the entire live process: the **Session Manager**.

Think of the Session Manager as the **conductor of an orchestra**, or the **director of a play**. It doesn't play any instruments or act in the play itself, but it tells everyone else what to do and when to do it, making sure the whole performance runs smoothly from start to finish.

## What is the Session Manager?

The Session Manager (`SessionManager` class in `src/session/session_manager.py`) is the heart of the *live monitoring loop*. Its main job is to orchestrate the continuous process of:

1.  Getting a picture (frame) from the camera.
2.  Sending that picture to different parts of the system for analysis (like figuring out where someone is looking or if there's a book).
3.  Taking the results from the analysis and combining them to figure out attention.
4.  Showing the picture on the screen, perhaps with some results drawn on it.
5.  Repeating these steps many times per second!

It ties together the [Camera Manager](03_camera_manager_.md), [Gaze Estimator](04_gaze_estimator_.md), [Book Detector](05_book_detector_.md), and [Attention Monitor](06_attention_monitor_.md) components, making them work together in a synchronized way.

## Why Do We Need a Session Manager?

Imagine you have a bunch of specialized robots: one gets images, one analyzes faces, one looks for books, and one draws on images. You need a central controller to tell them: "Okay, Robot Getter, get frame 1. Robot Face, analyze frame 1. Robot Book, analyze frame 1. Robot Drawer, draw results on frame 1. Now, Robot Getter, get frame 2..."

The Session Manager is this central controller. It ensures that the frames flow through the system correctly and that the different analysis steps happen in the right order, allowing the application to run continuously and provide real-time attention monitoring.

## How `main.py` Starts the Session

As we saw in the last chapter, `main.py` is responsible for creating the `SessionManager` object and telling it to start. Let's look at that part of the code again from `main.py`:

```python
# From main.py
# ... import statements and setup ...
from src.session.session_manager import SessionManager
# ... other component initializations (camera_manager, gaze_estimator) ...

def main():
    try:
        logger.info("Starting Book Attention Monitoring System")

        # ... create camera_manager (from Chapter 3) ...
        # ... create gaze_estimator (from Chapter 4) ...

        logger.info("Initializing session manager")
        session_manager = SessionManager( # Create the Session Manager object
            camera_manager=camera_manager, # Give it the camera manager
            gaze_analyzer=gaze_estimator, # Give it the gaze estimator
            book_detector_model_path=YOLO_MODEL_PATH # Tell it where the book model is
        )

        # Run the session
        logger.info("Starting attention monitoring session")
        session_manager.run_session() # <<< This is the command!
        # ... rest of main (cleanup) ...

```

This snippet shows that `main.py` first creates instances (objects) of the [Camera Manager](03_camera_manager_.md) and [Gaze Estimator](04_gaze_estimator_.md). Then, it creates the `SessionManager`, *giving* it those other objects. This is like giving the conductor the sheet music (model paths) and introducing them to the main musicians (camera, gaze estimator). Finally, `main.py` calls `session_manager.run_session()`. This is the crucial step – it tells the Session Manager: "Okay, you're in charge now, start the main show!"

## What Happens Inside `SessionManager.run_session()`?

When `run_session()` is called, the Session Manager gets to work. It essentially enters a loop that continues until you tell the program to stop (like by pressing the 'q' key on your keyboard).

Here's a simplified sequence of events within that loop:

```mermaid
sequenceDiagram
    participant SM as Session Manager
    participant CM as Camera Manager
    participant Analysis as Processing Thread
    participant Display as Display Window

    SM->>CM: 1. Start Camera
    loop Session Running
        SM->>CM: 2. Capture Frame
        CM-->>SM: Frame
        SM->>Analysis: 3. Queue Frame for Processing
        Analysis-->>SM: (Happens in background)
        SM->>Display: 4. Show latest processed frame
        Display-->>SM: User Input (e.g., 'q')
        alt User pressed 'q'
            SM->>SM: Stop Session Loop
            break
        end
    end
    SM->>CM: 5. Release Camera
    SM->>Display: 6. Close Display Window
    SM-->>MainPy: Session Ended
```

Let's break down these steps:

1.  **Start Camera:** The Session Manager first tells the [Camera Manager](03_camera_manager_.md) to start capturing frames.
2.  **Capture Frame:** It continuously asks the [Camera Manager](03_camera_manager_.md) for the latest available frame.
3.  **Queue Frame for Processing:** Instead of doing all the analysis directly in the main loop (which would slow down capturing frames), the Session Manager puts the captured frame into a queue. A separate "processing thread" (like a dedicated helper) is constantly watching this queue and takes frames out to analyze them using the [Gaze Estimator](04_gaze_estimator_.md), [Book Detector](05_book_detector_.md), and [Attention Monitor](06_attention_monitor_.md). This keeps the frame rate smooth.
4.  **Show Frame:** The Session Manager displays the *latest available processed frame* on the screen. This is where you see the video feed with boxes around books, gaze indicators, and attention status messages.
5.  **Check for Quit:** It constantly checks if the user has pressed a specific key (like 'q') to signal that they want to stop the session.
6.  **Loop or Stop:** If 'q' wasn't pressed, it goes back to step 2, getting the next frame. If 'q' *was* pressed, it breaks out of the loop.
7.  **Cleanup:** Once the loop finishes, the Session Manager tells the [Camera Manager](03_camera_manager_.md) to stop and release the camera, and closes the display windows.

This continuous cycle is why you see a live video feed and updated attention status while the program is running.

## Looking at the Session Manager Code (`src/session/session_manager.py`)

Let's peek at the core parts of the `SessionManager` class to see how it does this.

First, the `__init__` method sets everything up:

```python
# From src/session/session_manager.py
import logging
import threading # Used for running analysis in the background
from queue import Queue # Used to pass frames to the background thread
import cv2 # For displaying frames
import time # For timing

# ... import other components ...
from src.camera.camera_manager import CameraManager
from src.models.gaze_estimator import GazeEstimator
from src.models.book_detector import BookDetector # SessionManager creates this
from src.analysis.attention_monitor import AttentionMonitor # SessionManager creates this

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, camera_manager: CameraManager, gaze_analyzer: GazeEstimator, book_detector_model_path: str):
        """
        Initialize the session manager.
        Receives CameraManager and GazeEstimator, creates BookDetector and AttentionMonitor.
        """
        self.camera_manager = camera_manager # Received from main.py
        self.gaze_analyzer = gaze_analyzer   # Received from main.py

        # The Session Manager creates the Book Detector and Attention Monitor itself
        self.book_detector = BookDetector(book_detector_model_path)
        self.attention_monitor = AttentionMonitor() # Needs book model path too, but simplified here

        self.running = False # Flag to control the main loop
        self.frame_queue = Queue(maxsize=2) # A queue to hold frames for processing thread

        # ... other setup like variables for attention data, FPS, etc. ...

        logger.info("Session Manager initialized")

```

*   **Imports:** It imports the tools it needs, including the other components, logging, and tools for managing background tasks (`threading`, `Queue`).
*   **`__init__(...)`:** This method runs when the `SessionManager` object is created (by `main.py`). It receives the `camera_manager` and `gaze_analyzer` objects. It then *creates* the `BookDetector` and `AttentionMonitor` objects itself, giving the `BookDetector` the path to the model file it needs.
*   **`self.running`:** This is a simple flag (a True/False variable) that controls whether the main loop should keep running.
*   **`self.frame_queue`:** This creates a `Queue`. Think of it as a small box where the main loop puts frames, and the background processing thread takes them out. The `maxsize=2` means it can only hold 2 frames at a time, preventing it from using too much memory if processing is slow.

Now, let's look at the main loop structure in the `run_session` method:

```python
# From src/session/session_manager.py
    def run_session(self) -> None:
        """Run the attention monitoring session"""
        try:
            logger.info("Starting camera capture session")
            self.camera_manager.start() # Tell camera to start capturing

            self.running = True
            # Start the background thread for processing frames
            process_thread = threading.Thread(target=self._process_frames)
            process_thread.daemon = True # Daemon threads stop when the main program stops
            process_thread.start() # Start the thread

            logger.info("Entering main session loop (display loop)")

            # This loop runs continuously to capture and display frames
            while self.running: 
                
                # 1. Get a frame from the camera
                frame = self.camera_manager.capture_frame() 
                if frame is None:
                    time.sleep(0.01) # Wait a bit if no frame was ready
                    continue # Skip the rest of this loop turn and try again

                # 2. Put the frame in the queue for the background thread
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy()) # Use .copy() to be safe with threads

                # 3. Display the latest frame (might be from a bit ago, but keeps display smooth)
                self._display_frame(frame) # This method handles showing the window

                # 4. Check if user pressed 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Session ended by user")
                    self.running = False # Set flag to stop the while loop
            
        except Exception as e:
            logger.error(f"Unexpected error in session: {str(e)}", exc_info=True)
        finally:
            # This code runs after the while loop stops (either normally or due to error)
            self._cleanup(process_thread) # Call the cleanup method

```

*   **`self.camera_manager.start()`:** Initializes the camera.
*   **`process_thread = threading.Thread(...)`:** This line sets up a new, separate process (a "thread") that will run the `_process_frames` method in the background. This is important because analyzing frames (gaze, book detection) takes time. If it happened in the main loop, the loop would pause, and the video display would look choppy. By putting it in a separate thread, the main loop can keep getting frames and updating the display smoothly while the analysis happens whenever the background thread is ready.
*   **`while self.running:`:** This is the main loop. It will keep going as long as the `self.running` variable is `True`.
*   **`self.camera_manager.capture_frame()`:** Gets a single frame from the camera feed.
*   **`self.frame_queue.put(...)`:** Adds the captured frame to the queue, where the background `_process_frames` thread will eventually pick it up.
*   **`self._display_frame(frame)`:** Calls another method to show the frame using OpenCV (`cv2`). This method also adds overlays like FPS or attention status using the results calculated by the background thread.
*   **`cv2.waitKey(1) & 0xFF == ord('q')`:** This is a standard way in OpenCV to check for a key press. It waits for 1 millisecond (so it doesn't freeze the program) and sees if the 'q' key was pressed.
*   **`self.running = False`:** If 'q' is pressed, this line changes the flag, which causes the `while self.running:` loop to finish after the current turn.
*   **`finally: self._cleanup(process_thread)`:** The code in the `finally` block runs when the `try` block finishes (either normally by setting `self.running` to `False` or if an error happens). It calls the `_cleanup` method to stop the background thread, release the camera, and close windows.

The `_process_frames` method (not shown in detail here for simplicity) is where the frame is taken from the queue and passed to `self.gaze_analyzer.estimate_gaze()`, `self.book_detector.detect_objects()`, and finally `self.attention_monitor.analyze_attention()`. The results are stored in `self.last_attention_data` and `self.last_processed_frame` so the display loop can use them.

The `_display_frame` method (also not shown in detail) takes the latest captured frame and the results from `self.last_attention_data` and `self.last_processed_frame` to create the image you see in the window, drawing boxes, text, etc., before calling `cv2.imshow` to put it on the screen.

## Conclusion

The `SessionManager` is the core conductor of our application's live monitoring process. It's initialized by `main.py` and then takes control, managing the continuous cycle of capturing frames, sending them off for analysis (using separate threads for efficiency), displaying the results, and listening for the user to stop the session.

It doesn't perform gaze estimation or book detection itself, but it directs the components that do, ensuring they work together seamlessly to monitor attention in real-time.

Now that we understand the central loop managed by the Session Manager, let's look at the first specialized component it relies on: the [Camera Manager](03_camera_manager_.md), which is responsible for getting those precious video frames in the first place!

[Next Chapter: Camera Manager](03_camera_manager_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)