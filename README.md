
# Persian Character Recognizer (Hebb Rule)

This project demonstrates a basic Persian character recognition system using a Hebbian Artificial Neural Network (ANN). It features a PyQt5-based graphical user interface (GUI) that lets users draw characters with a mouse or a graphics tablet (like an XP-Pen Deco V2), train the neural network with these drawings, and then recognize new ones.

Currently, the system is configured to distinguish between two specific Persian characters: هـ (Heh) and ج (Jeem), as well as a general "Other/Unrecognized" category. This offers a hands-on introduction to fundamental neural network concepts and their application in image recognition.

## Features

- **Interactive Drawing Canvas:** Draw characters directly on the PyQt5 GUI.
- **Tablet Support:** Optimized for XP-Pen Deco V2 (and potentially other tablets) with pressure sensitivity.
- **Image Preprocessing:** Automatically normalizes and centers drawn characters to a 28x28 pixel grayscale format, suitable for ANN input.
- **Hebbian ANN Implementation:** A custom, from-scratch implementation of a multi-class Hebbian neural network.
- **Interactive Training:** Train the network by categorizing your drawings as 'هـ', 'ج', or 'Other'.
- **Real-time Recognition:** Get instant predictions on new drawings after training.
- **Visual Feedback:** See the normalized 28x28 input image used by the ANN.

## Getting Started

### Prerequisites

You'll need Python 3 and the following libraries:

- `PyQt5`: For the graphical user interface.
- `NumPy`: For numerical operations, especially with array manipulation for image data and network weights.

You can install them using pip:

```bash
  pip install PyQt5 numpy
```

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/haadijafari/Alphabet-Recognition.git
cd persian-char-recognizer-hebb
```

**2. Ensure file structure:**
\
Make sure `drawing_input_manager.py` and `hebbian_ann_recognizer.py` are in the same directory.

### Running the Application

Execute the main application script:

```bash
python3 hebbian_ann_recognizer.py
```

## Usage and Training

When the application window appears, you'll see a drawing canvas and several buttons.

**1. Clear Drawing: Clears the canvas.**

**2. Training the Network (Crucial for Performance!):**

The network starts with some very basic "dummy" training patterns. For accurate recognition, **you must train the network with your own drawings.**

- **Draw 'هـ':** Use your mouse or tablet to draw a clear example of the Persian letter هـ. Then, click the "Train as 'هـ'" button.
- **Draw 'ج':** Clear the canvas. Draw a clear example of the Persian letter ج. Then, click the "Train as 'ج'" button.
- **Draw 'Other/Noise':** Clear the canvas. Draw something that is neither 'هـ' nor 'ج' (e.g., a random scribble, another letter, or simply leave the canvas blank). Then, click the "Train as 'Other/Noise'" button.
**Repeat these training steps multiple times (at least 5-10 examples for each category is recommended).** The more diverse and consistent your training examples are, the better the network will learn to differentiate. Each time you train, the network's weights are adjusted based on all the patterns it has learned so far.

**3. Recognize Character:**

After training, clear the canvas. Draw a new character (either 'هـ', 'ج', or something else). Click the **"Recognize Character"** button. The application will display:

- A preview of the normalized 28x28 image that the ANN processed.
- The recognized character (e.g., "It's 'هـ'", "It's 'ج'", or "Other/Unrecognized").
- A "Confidence" score, which is the raw activation value of the winning output neuron.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests. Suggestions for improvements, especially regarding more advanced ANN architectures or dataset integration, are welcome!

## License
This project is open-source and available under the [MIT LICENSE](https://choosealicense.com/licenses/mit/). See the file for more details.
