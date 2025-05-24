# Packages and libraries used throughout this project
import os                         # For environment variables
import cv2                        # OpenCV for camera and image processing
import mediapipe as mp            # Google's framework for hand tracking
from collections import Counter   # To count most common finger counts
from enum import Enum, auto       # For managing stages of input

# Disable unnecessary TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Stage(Enum):
    """
    Enumeration for different stages of calculator input.
    """

    IDLE = auto()        # Waiting for "0 fingers" gesture to start addition
    INPUT_ONE = auto()   # Reading first number input
    INPUT_TWO = auto()   # Reading second number input
    READY = auto()       # Both inputs ready, wait for calculation trigger


class HandCalculator:
    """
    A gesture-based calculator that uses hand landmarks to perform addition
    of two numbers (1–5) shown using fingers on a webcam feed.
    """

    # Landmark indices used by MediaPipe for finger detection
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    THUMB_TIP = 4
    THUMB_IP = 3

    def __init__(self, camera_index=0):
        """
        Initializes the calculator, opens webcam, and sets up MediaPipe.
        """
        self.stage = Stage.IDLE
        self.buffer = []
        self.operand1 = None
        self.operand2 = None

        self.cap = cv2.VideoCapture(camera_index)  # Open camera
        self.window_name = 'Finger Calculator'

        # Initialize MediaPipe Hands
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Verify camera availability
        if not self.cap.isOpened():
            raise IOError('Could not open camera.')

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def run(self):
        # Main loop: reads frames, processes hand data, updates state, and renders output
        try:
            while True:
                frame = self._read_frame()
                finger_count = self._detect_fingers(frame)
                self._update_state(finger_count)
                self._render(frame, finger_count)
                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key):
                    break
        finally:
            self._cleanup()

    def _read_frame(self):
        """
        Captures and returns a horizontally flipped frame from the webcam.
        """
        success, frame = self.cap.read()
        if not success:
            raise RuntimeError('Could not read frame from camera.')
        return cv2.flip(frame, 1)  # Flip horizontally for natural interaction

    def _detect_fingers(self, frame):
        """
        Detects the number of fingers raised using MediaPipe landmarks.

        Parameters:
            frame (numpy.ndarray): The current webcam frame.

        Returns:
            int: Number of fingers detected as being raised.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return 0

        lm = results.multi_hand_landmarks[0].landmark
        count = 0

        # Thumb detection using x-coordinate comparison
        if lm[self.THUMB_TIP].x < lm[self.THUMB_IP].x:
            count += 1

        # Other fingers: tip above pip (y-coordinate)
        for tip, pip in zip(self.FINGER_TIPS, self.FINGER_PIPS):
            if lm[tip].y < lm[pip].y:
                count += 1

        # Draw landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp.solutions.hands.HAND_CONNECTIONS
        )

        return count

    def _update_state(self, count):
        """
        Updates the calculator's stage based on the number of fingers detected.

        Parameters:
            count (int): Number of raised fingers.
        """

        if self.stage == Stage.IDLE:
            if count == 0:
                self._transition(Stage.INPUT_ONE)

        elif self.stage in (Stage.INPUT_ONE, Stage.INPUT_TWO):
            if count > 0:
                self.buffer.append(count)
            elif self.buffer:
                # Choose the most common count (most stable gesture)
                most_common = Counter(self.buffer).most_common(1)[0][0]
                if self.stage == Stage.INPUT_ONE:
                    self.operand1 = most_common
                    self._transition(Stage.INPUT_TWO)
                else:
                    self.operand2 = most_common
                    self._transition(Stage.READY)
                self.buffer.clear()

    def _transition(self, new_stage):
        """
        Changes the current input stage.

        Parameters:
            new_stage (Stage): The new stage to transition to.
        """
        self.stage = new_stage

    def _render(self, frame, count):
        """
        Overlays the current status, detected counts, and inputs on the frame.

        Parameters:
            frame (numpy.ndarray): The current webcam frame.
            count (int): Number of fingers detected.
        """
        hints = {
            Stage.IDLE: 'Show 0 fingers to start +',
            Stage.INPUT_ONE: 'Show number 1',
            Stage.INPUT_TWO: 'Show number 2',
            Stage.READY: "Press 's' for result",
        }

        cv2.putText(frame, hints[self.stage], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if count > 0 and self.stage in (Stage.INPUT_ONE, Stage.INPUT_TWO):
            cv2.putText(frame, f'Current: {count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if self.operand1 is not None:
            cv2.putText(frame, f'1: {self.operand1}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if self.operand2 is not None:
            cv2.putText(frame, f'2: {self.operand2}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow(self.window_name, frame)

    def _handle_key(self, key):
        """
        Handles key input from the user.

        Parameters:
            key (int): Key code captured by OpenCV.

        Returns:
            bool: True if the program should exit, False otherwise.
        """
        if key in (27, ord('e')):
            return True

        if key == ord('s') and self.stage == Stage.READY:
            result = self.operand1 + self.operand2
            print(f'Calculation: {self.operand1} + {self.operand2} = {result}')
            self.operand1 = None
            self.operand2 = None
            self.buffer.clear()
            self._transition(Stage.IDLE)

        return False

    def _cleanup(self):
        """
            Releases system resources (camera and window) on exit.
        """
        # Releases the camera and destroys all windows
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    calc = HandCalculator()
    calc.run()
