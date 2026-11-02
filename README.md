
# Persian Character Recognizer (Hebb Rule)

This project demonstrates a basic Persian character recognition system using single-layer Artificial Neural Networks (ANNs), built around the Hebbian learning rule. It features a PyQt5-based graphical user interface (GUI) that lets users draw characters with a mouse or a graphics tablet (like an XP-Pen Deco V2), train the neural network with these drawings, and then recognize new ones. Besides the Hebb rule, the network can be switched to the Perceptron rule or the Delta rule (Adaline) to compare them on the same training data.

Currently, the system is configured to distinguish between two specific Persian characters: هـ (Heh) and ج (Jeem), as well as a general "Other/Unrecognized" category. This offers a hands-on introduction to fundamental neural network concepts and their application in image recognition.

## Features

- **Interactive Drawing Canvas:** Draw characters directly on the PyQt5 GUI.
- **Tablet Support:** Optimized for XP-Pen Deco V2 (and potentially other tablets) with pressure sensitivity. The stylus eraser end erases.
- **Image Preprocessing:** Automatically normalizes and centers drawn characters to a 28x28 pixel grayscale format, suitable for ANN input.
- **Hebbian ANN Implementation:** A custom, from-scratch implementation of a multi-class Hebbian neural network.
- **Switchable Learning Algorithms:** Switch between the Hebb rule, the Perceptron rule and the Delta rule (Adaline) from a drop-down; the new algorithm is retrained on the same patterns.
- **Interactive Training:** Train the network by categorizing your drawings as 'هـ', 'ج', or 'Other'.
- **Saved Training Data:** Every training pattern is saved to disk and loaded again on the next launch.
- **Undo:** Remove the last training pattern if you clicked the wrong button.
- **Real-time Recognition:** Get instant predictions on new drawings after training.
- **Visual Feedback:** See the normalized 28x28 input image used by the ANN, the number of saved patterns per class, and the training accuracy of the current algorithm.

## Getting Started

### Prerequisites

You'll need Python 3.10 or newer and the following libraries:

- `PyQt5`: For the graphical user interface.
- `NumPy`: For numerical operations, especially with array manipulation for image data and network weights.

The project uses [uv](https://docs.astral.sh/uv/) to manage its virtual environment and dependencies (see `pyproject.toml` and `uv.lock`).

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/haadijafari/Alphabet-Recognition.git
cd Alphabet-Recognition
```

**2. Create the virtual environment and install the dependencies:**

```bash
uv sync
```

This creates a `.venv` directory with the exact versions pinned in `uv.lock`.

<details>
<summary>Without uv (pip)</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

</details>

### Running the Application

Execute the main application script:

```bash
uv run hebbian_ann_recognizer.py
```

(With pip, activate the virtual environment and run `python3 hebbian_ann_recognizer.py`.)

Command-line options:

| Option | Description |
|---|---|
| `--algorithm {hebbian,perceptron,adaline}` | Learning algorithm to start with (default: `hebbian`). It can still be switched in the window. |
| `--data-dir DIR` | Directory for saved training patterns (default: `training_data/` next to the script). |

For example:

```bash
uv run hebbian_ann_recognizer.py --algorithm perceptron
```

To try only the drawing canvas and image normalization, run `uv run drawing_input_manager.py`.

## Usage and Training

When the application window appears, you'll see a drawing canvas and several buttons.

**1. Clear Drawing: Clears the canvas.**

**2. Training the Network (Crucial for Performance!):**

Until you save your own patterns, the network uses some very basic built-in "starter" patterns. For accurate recognition, **you must train the network with your own drawings.**

- **Draw 'هـ':** Use your mouse or tablet to draw a clear example of the Persian letter هـ. Then, click the "Train as 'هـ'" button.
- **Draw 'ج':** Draw a clear example of the Persian letter ج. Then, click the "Train as 'ج'" button.
- **Draw 'Other/Noise':** Draw something that is neither 'هـ' nor 'ج' (e.g., a random scribble or another letter). Then, click the "Train as 'Other/Noise'" button.

The canvas is cleared automatically after each training click. Training on a blank canvas is not allowed.

**Repeat these training steps multiple times (at least 5-10 examples for each category is recommended).** The more diverse and consistent your training examples are, the better the network will learn to differentiate. Each time you train, the network is retrained on all the patterns saved so far.

**3. Undo a Wrong Pattern:**

Click **"Undo Last Training Pattern"** to remove the most recently added pattern (and its saved file) and retrain. This also works after restarting the application.

**4. Recognize Character:**

Draw a new character (either 'هـ', 'ج', or something else). Click the **"Recognize Character"** button. The application will display:

- A preview of the normalized 28x28 image that the ANN processed.
- The recognized character ('هـ', 'ج', or "Other/Unrecognized").
- A "Score" for the winning output neuron, e.g. `Recognition Result: هـ (Score: 0.52)`. Its meaning and range depend on the algorithm; see [Understanding the Score](#understanding-the-score).

## Learning Algorithms

All three algorithms use the same single-layer network: 784 inputs (one per pixel, 0 = background, 1 = ink) and 2 output neurons (one for 'هـ', one for 'ج'). The target vectors are bipolar: `[1, -1]` for 'هـ', `[-1, 1]` for 'ج' and `[-1, -1]` for 'Other'. The implementations are in `learning_algorithms.py`.

| Algorithm | Learning rule | Recognized when |
|---|---|---|
| **Hebbian** (default) | One pass: every pattern adds `x · t` to the weights, whether or not it is already classified correctly. | The best cosine similarity between the input and a neuron's weights (in [-1, 1]) is at least 0.1. |
| **Perceptron** | Weights change only when a neuron's output is wrong. Repeats over the data until a pass has no mistakes (at most 100 passes). | The best neuron's activation is above 0. |
| **Delta rule (Adaline)** | 50 passes that minimize the squared error between the activation and the target (normalized LMS). | The best neuron's activation is above 0. |

If no neuron passes its threshold, the drawing is shown as "Other/Unrecognized". The scores and thresholds are explained in detail in [Understanding the Score](#understanding-the-score).

Switching the algorithm in the drop-down retrains it on the same saved patterns, and the status line shows its **training accuracy**: how many saved patterns it classifies correctly. This makes it easy to compare the algorithms on the same data. Because the Hebb rule never corrects its mistakes, it usually fits the training data less well than the Perceptron and Delta rules. Training uses a fixed random seed, so the same data always gives the same weights.

## Understanding the Score

### How a drawing is classified

Each of the two output neurons gets a score for the drawing: neuron 1 for 'هـ' and neuron 2 for 'ج'. Recognition then works the same way for every algorithm:

1. The neuron with the **highest score** is the candidate.
2. If its score is **at least the algorithm's threshold**, the drawing is recognized as that neuron's letter.
3. Otherwise, neither letter matches well enough and the result is **"Other/Unrecognized"**.

The **Score** shown in the result is always the candidate's (highest) score, even when the result is "Other/Unrecognized". In that case it is the best score, which was still below the threshold. For example, `Other/Unrecognized (Score: 0.04)` with the Hebbian algorithm means the closest letter reached only 0.04, below the 0.1 threshold.

### What the score means for each algorithm

| | Hebbian | Perceptron | Delta rule (Adaline) |
|---|---|---|---|
| **Score is** | Cosine similarity between the drawing and the neuron's weights | Raw activation: weighted sum of the inputs plus bias | Raw activation: weighted sum of the inputs plus bias |
| **Range** | -1 to 1 | Unbounded | Unbounded, usually around -1 to 1 |
| **Threshold** | 0.1 | 0 | 0 |
| **Typical good match** | About 0.3 to 0.7 | Any clearly positive value | Close to 1 |

**Hebbian: cosine similarity.** The score measures how much the drawing's ink overlaps with the pixels the neuron has learned to associate with its letter, independent of how much ink there is:

- **1** would mean the drawing has exactly the same shape as the neuron's weight pattern. In practice this does not happen, because each neuron's weights are the sum of its own letter's patterns *minus* the patterns of the other classes, so good matches usually score about 0.3 to 0.7.
- **Around 0** means the drawing has little in common with what the neuron learned.
- **Negative** means the ink falls mostly on pixels that belong to the *other* classes.

The raw Hebbian activation is not used as the score. It grows with the number of training patterns and the amount of ink, so a fixed threshold on it would mean something different after every training click. The cosine similarity stays on the same -1 to 1 scale no matter how much training data there is.

**Perceptron: raw activation.** Only the **sign** matters for the decision: above 0 means the neuron says "this is my letter", below 0 means "it is not". The size of the number tells you roughly how far the drawing is from the neuron's decision boundary, but its scale depends on how many corrections were made during training. Every mistake makes the weights larger. So a score of 5 after one training run and 5 after another do not mean the same thing. Compare only the sign, or scores from the same trained network.

**Delta rule (Adaline): raw activation trained toward the targets.** The Delta rule trains each neuron's activation to be exactly **+1** for its own letter and **-1** for everything else. This makes the score the easiest to read:

- **About 1:** looks like a typical training example of this letter.
- **About -1:** clearly not this letter.
- **Near 0:** uncertain; the drawing is between the classes.
- **Above 1:** possible for drawings with more ink than the training examples (for example, thicker strokes). Still a match.

### What the score is not

- **Not a probability or a percentage.** A Hebbian score of 0.6 does not mean "60% sure".
- **Not comparable between algorithms.** A Hebbian 0.3 and a Delta rule 0.3 mean different things: the first is a fairly good match, the second is an uncertain one.
- **Not a measure of handwriting quality.** It only says how close the drawing is to *your* saved training patterns.

### Seeing all the scores

The result shows only the winning neuron's score. The terminal prints both neurons' values for every recognition:

```
Activations: [ 4.64 -4.64], scores: [ 0.088 -0.088]
```

The first number in each list is the 'هـ' neuron and the second is the 'ج' neuron. `Activations` are the raw weighted sums and `scores` are the values compared with the threshold. For the Perceptron and the Delta rule both lists are the same; for Hebbian, `scores` are the cosine similarities.

### Adjusting the threshold

The thresholds are the `decision_threshold` class attributes in `learning_algorithms.py`. If correct letters are often shown as "Other/Unrecognized", lower the threshold. If scribbles are often recognized as letters, raise it. Use the scores printed in the terminal to choose a value between the scores of your correct letters and your "Other" drawings. The Hebbian threshold (0.1) is the one most likely to need tuning. The Perceptron and Delta rule thresholds (0) follow from how they are trained and normally stay at 0.

## Saved Training Data

Each training pattern is saved as a 28x28 grayscale PNG:

```
training_data/
├── heh/     20261102-143012-123456.png ...
├── jeem/
└── other/
```

The folder name is the pattern's class and the filename is the time it was added, which is the order used for undo. PNG is lossless, so a reloaded pattern is identical to the original. You can open the images to check your training data, delete a file to remove that pattern, or copy the folder to share a training set. Restart the application after changing files by hand.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests. Suggestions for improvements, especially regarding more advanced ANN architectures or dataset integration, are welcome!

## License
This project is open-source and available under the [MIT LICENSE](https://choosealicense.com/licenses/mit/). See the file for more details.
