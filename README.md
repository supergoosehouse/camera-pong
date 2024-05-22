# Camera Pong

![logo](templates/camera%20pong.png)

Camera Pong is a classic pong game with a twist - instead of using keyboard controls, it utilizes your webcam and hand recognition to control the pong racket. This adds an interactive and immersive element to the gameplay experience.

## Preview

![preview1](/preview/CPpreview1.gif)

## Technologies Used

- JavaScript (JS)
- Socket.IO
- Flask (Python)
- TensorFlow
- OpenCV (CV2)
- HTML
- CSS

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/supergoosehouse/camera-pong.git
   ```

2. Install dependencies:
   ```shell
   cd camera-pong
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask server:

   ```shell
   test_server.py
   ```

2. Open the game in your web browser:

   ```
   http://localhost:5000
   ```

3. Use hand gestures to control the pong racket and play the game!

## Features

- **Hand Tracking:** Utilizes hand tracking technology to detect and track the movement of the player's hand.
- **Gesture Control:** Enables controlling the pong racket using the index and middle finger tips, offering a unique and intuitive gameplay experience.

- **Interactive Gameplay:** Allows playing the classic pong game using hand gestures detected by the webcam, adding an interactive element to the gameplay.

- **Gesture Recognition:** Recognizes various hand gestures as such:
  - Peace
  - Rock
  - ThumbUp
  - ThumbDown
  - OK
  - PinchedFingers
