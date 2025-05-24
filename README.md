# Skills: Advanced Programming with Advanced Computer Languages

## Project Title: Hand Gesture Calculator

---

## Project Overview

This project demonstrates a touchless, gesture-based calculator that uses **computer vision** to detect hand signals (fingers) via a webcam. It performs simple addition of two numbers (from 1–5), using **MediaPipe** for hand landmark detection and **OpenCV** for image processing and display.

The calculator operates in real time: users input numbers by holding up a number of fingers, and the system sums them when prompted.

---

## Assignment Deliverables

The deliverables for this assignment include the following file:

- `main.py` – Source code for the Python program.

---

## Assignment Background

This project explores the practical application of gesture recognition using hand landmarks. The calculator operates in stages and uses MediaPipe’s hand tracking module to detect fingers, using that as numeric input for arithmetic operations.

The state machine works as follows:
- `IDLE`: Waits for the user to show **0 fingers**.
- `INPUT_ONE`: First number is recorded using raised fingers.
- `INPUT_TWO`: Second number is recorded similarly.
- `READY`: User presses `'s'` to compute the result.

The calculator uses `cv2.putText()` to display instructions, the current count, and operand values on the webcam feed.

---

## Assignment Specifications

### Required Features

1. **State Management**: Handled via an `Enum` (`Stage`) with transitions triggered by finger counts.
2. **Finger Detection**: Uses MediaPipe to identify the number of fingers raised.
3. **Buffered Input**: Finger counts are buffered and the most common is used to improve stability.
4. **Rendering**: Live instructions and feedback are drawn on the camera feed using OpenCV.
5. **User Input**:
   - Press `'s'`: Performs addition when in `READY` state.
   - Press `'e'` or `ESC`: Exits the program.

### Libraries Used

- `opencv-python`
- `mediapipe`
- `enum`, `collections`, `os`

---

## How It Works

### Input Flow:

1. **Start**: Show 0 fingers to begin input.
2. **First Number**: Raise 1–5 fingers; close hand to confirm.
3. **Second Number**: Repeat gesture for second operand.
4. **Result**: Press `'s'` to see the result in terminal.
5. **Exit**: Press `'e'` or `ESC`.

---

## Running the Program

This program can be run in **any Python IDE or console** that supports Python 3 and has the required libraries installed.

> **Important**: A working **webcam is required** to use the hand gesture functionality.

To run the program:

1. Ensure your webcam is connected and functional.
2. Install the required libraries:

```bash
pip install opencv-python mediapipe
