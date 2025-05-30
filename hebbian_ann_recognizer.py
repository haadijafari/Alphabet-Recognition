import numpy as np
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# Import the DrawingInputManager class from your local file.
# Make sure 'drawing_input_manager.py' is in the same directory as this script.
from drawing_input_manager import DrawingInputManager

class HebbianANN:
    """
    Implements a simple Artificial Neural Network based on the Hebbian learning rule.
    This network supports multi-class classification using multiple output neurons,
    where each neuron corresponds to a specific class.
    """
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 1.0):
        """
        Initializes the Hebbian neural network.

        Args:
            input_size (int): The number of input neurons. This should match the
                              flattened size of the input patterns (e.g., 28*28 = 784 for a 28x28 image).
            output_size (int): The number of output neurons. Each output neuron
                                typically represents a class to be recognized.
        """
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weights as a matrix: (input_size x output_size).
        # Each column in this matrix represents the weight vector for a single output neuron.
        self.weights = np.zeros((input_size, output_size))
        # The learning rate for Hebbian learning. For simple Hebbian, it's often 1.0.
        self.learning_rate = learning_rate

    def train(self, patterns: np.ndarray, targets: np.ndarray):
        """
        Trains the Hebbian network using the provided patterns and their corresponding target vectors.
        The Hebbian rule updates weights based on the co-occurrence of input and output activations:
        "Neurons that fire together, wire together."

        Args:
            patterns (np.ndarray): A 2D NumPy array where each row is a training pattern.
                                   Shape: (num_samples, input_size).
                                   Values must be bipolar (-1 or 1).
            targets (np.ndarray): A 2D NumPy array of corresponding target output vectors for each pattern.
                                  Shape: (num_samples, output_size).
                                  Each vector should contain bipolar values (-1 or 1).
        Raises:
            ValueError: If the dimensions of patterns or targets do not match the network's configuration.
        """
        # Validate input dimensions to prevent errors during training.
        if patterns.shape[1] != self.input_size:
            raise ValueError(f"Pattern input size {patterns.shape[1]} does not match network input size {self.input_size}")
        if patterns.shape[0] != targets.shape[0]:
            raise ValueError("Number of patterns and targets must be the same.")
        if targets.shape[1] != self.output_size:
            raise ValueError(f"Target output size {targets.shape[1]} does not match network output size {self.output_size}")

        print("Starting Hebbian training...")
        # Reset weights to zero before retraining. This is important for this specific
        # implementation of Hebbian learning, where weights accumulate over all patterns.
        self.weights = np.zeros((self.input_size, self.output_size))

        # Iterate through each training sample (pattern and its target vector).
        for i in range(patterns.shape[0]):
            x = patterns[i]          # Current input pattern (1D array).
            y_vector = targets[i]    # Current target output vector (1D array).

            # Update weights for each individual output neuron.
            # The Hebbian rule: delta_w_ij = learning_rate * x_i * y_j
            # Here, w_ij is the weight connecting input i to output neuron j.
            for j in range(self.output_size):
                # Update the weights corresponding to the j-th output neuron.
                # 'self.weights[:, j]' refers to all weights connected to the j-th output neuron.
                self.weights[:, j] += self.learning_rate * x * y_vector[j]
            print(f"  Trained with pattern {i+1}. Target: {y_vector}")
        print("Training complete.")
        # Print the sum of all weights for a quick check that training occurred.
        print(f"Final weights sum: {np.sum(self.weights)}")

    def predict(self, input_pattern: np.ndarray) -> np.ndarray:
        """
        Predicts the activation levels for each output neuron for a given input pattern.
        In a simple Hebbian network, this involves calculating the weighted sum of inputs.

        Args:
            input_pattern (np.ndarray): A 1D NumPy array representing the input pattern.
                                        Values must be bipolar (-1 or 1).

        Returns:
            np.ndarray: A 1D NumPy array of activations (raw weighted sums) for each output neuron.
                        The length of this array equals `self.output_size`.
        Raises:
            ValueError: If the input pattern's size does not match the network's input size.
        """
        # Validate input pattern size.
        if len(input_pattern) != self.input_size:
            raise ValueError(f"Input pattern size {len(input_pattern)} does not match network input size {self.input_size}")

        # Calculate the weighted sum for each output neuron.
        # np.dot(input_pattern, self.weights) efficiently performs this:
        # For each output neuron 'j', it calculates sum(input_pattern[i] * self.weights[i, j])
        # The result is a 1D array where each element is the net input for one output neuron.
        net_inputs = np.dot(input_pattern, self.weights)

        # For Hebbian networks, these raw activations are often directly used for classification
        # (e.g., choosing the output with the highest activation).
        return net_inputs


class RecognitionApp(QWidget):
    """
    A PyQt5 application that integrates the DrawingInputManager with a HebbianANN
    to allow users to draw characters, train the network with examples,
    and then recognize new drawings.
    """
    def __init__(self):
        """
        Initializes the recognition application, sets up the UI,
        and prepares the Hebbian ANN for training and prediction.
        """
        super().__init__()
        self.setWindowTitle("Persian Character Recognizer (هـ, ج)")
        # Set the initial window size and position.
        self.setGeometry(100, 100, 500, 750)

        # Define canvas and ANN input resolution.
        self.canvas_size = (300, 300)
        self.ann_input_res = (28, 28) # Standard resolution for MNIST-like inputs.
        self.input_neuron_count = self.ann_input_res[0] * self.ann_input_res[1] # 28 * 28 = 784.
        self.num_classes = 2 # Recognizing 'هـ' and 'ج'.

        # Define target vectors for each class. These are bipolar.
        # The order of elements in the vector corresponds to the output neuron order.
        # E.g., for 'هـ', the first neuron is active (1), the second is inactive (-1).
        self.class_map = {
            'heh': np.array([1, -1]),   # Target vector for 'هـ'
            'jeem': np.array([-1, 1]),  # Target vector for 'ج'
            'other': np.array([-1, -1]) # Target vector for patterns that are neither 'هـ' nor 'ج'
        }
        # A mapping from output neuron index to human-readable class name.
        self.class_names = {
            0: 'هـ', # Index 0 corresponds to the first output neuron in the target vector.
            1: 'ج'   # Index 1 corresponds to the second output neuron.
        }

        # Initialize the Hebbian neural network with the determined input and output sizes.
        self.hebbian_net = HebbianANN(input_size=self.input_neuron_count, output_size=self.num_classes)
        # Lists to store collected training patterns and their corresponding target vectors.
        self.training_patterns = []
        self.training_targets = []

        # Set up the graphical user interface.
        self._init_ui()
        # Load initial dummy training data and train the network.
        # This provides a basic starting point but should be augmented by user training.
        self._load_and_train_network()

    def _init_ui(self):
        """
        Sets up the layout and widgets for the application window.
        Includes the drawing canvas, clear button, training buttons,
        recognition button, and result display.
        """
        main_layout = QVBoxLayout() # Main vertical layout for the entire window.

        # Create and add the drawing canvas widget.
        self.drawing_manager = DrawingInputManager(parent=self,
                                                 canvas_size=self.canvas_size,
                                                 normalized_size=self.ann_input_res)
        main_layout.addWidget(self.drawing_manager)

        # Button to clear the drawing canvas.
        clear_button = QPushButton("Clear Drawing")
        clear_button.clicked.connect(self.drawing_manager.clear_canvas)
        main_layout.addWidget(clear_button)

        # Horizontal layout for training buttons.
        training_buttons_layout = QHBoxLayout()

        # Button to train the current drawing as 'هـ'.
        train_heh_button = QPushButton("Train as 'هـ'")
        # Connect to a lambda function to pass the specific target vector for 'هـ'.
        train_heh_button.clicked.connect(lambda: self._add_training_pattern(self.class_map['heh']))
        training_buttons_layout.addWidget(train_heh_button)

        # Button to train the current drawing as 'ج'.
        train_jeem_button = QPushButton("Train as 'ج'")
        # Connect to a lambda function to pass the specific target vector for 'ج'.
        train_jeem_button.clicked.connect(lambda: self._add_training_pattern(self.class_map['jeem']))
        training_buttons_layout.addWidget(train_jeem_button)

        # Button to train the current drawing as 'Other/Noise' (neither 'هـ' nor 'ج').
        train_other_button = QPushButton("Train as 'Other/Noise'")
        # Connect to a lambda function to pass the specific target vector for 'other'.
        train_other_button.clicked.connect(lambda: self._add_training_pattern(self.class_map['other']))
        training_buttons_layout.addWidget(train_other_button)

        # Add the horizontal layout of training buttons to the main layout.
        main_layout.addLayout(training_buttons_layout)

        # Visual separator for clarity between training and recognition sections.
        main_layout.addWidget(QLabel("--- Recognition ---"))

        # Button to trigger the recognition process for the current drawing.
        process_button = QPushButton("Recognize Character")
        process_button.clicked.connect(self._process_and_recognize)
        main_layout.addWidget(process_button)

        # Label to display the recognition result.
        self.result_label = QLabel("Recognition Result: Draw and click 'Recognize'")
        self.result_label.setAlignment(Qt.AlignCenter) # Center the text.
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: blue;") # Style the text.
        main_layout.addWidget(self.result_label)

        # Label to display a preview of the normalized input image.
        self.normalized_display_label = QLabel("Normalized Input Preview:")
        self.normalized_display_label.setAlignment(Qt.AlignCenter) # Center the label.
        main_layout.addWidget(self.normalized_display_label)

        # Set the main layout for the QWidget.
        self.setLayout(main_layout)

    def _convert_to_bipolar(self, numpy_array_0_1: np.ndarray) -> np.ndarray:
        """
        Converts a NumPy array of pixel values from the unipolar range [0, 1]
        (where 0 is white, 1 is black) to a bipolar range [-1, 1].
        This conversion is typical for Hebbian networks.

        Args:
            numpy_array_0_1 (np.ndarray): A NumPy array with values between 0.0 and 1.0.

        Returns:
            np.ndarray: A NumPy array with values between -1.0 and 1.0.
                        0.0 (white) becomes -1.0, and 1.0 (black) becomes 1.0.
        """
        return 2 * numpy_array_0_1 - 1

    def _add_training_pattern(self, target_vector: np.ndarray):
        """
        Captures the current drawing from the `DrawingInputManager`,
        converts it into a bipolar NumPy array, and adds it along with its
        specified target vector to the training dataset. After adding, it
        clears the canvas and retrains the network.

        Args:
            target_vector (np.ndarray): The bipolar target vector (e.g., [1, -1] for 'هـ')
                                        corresponding to the current drawing.
        """
        # Get the normalized QImage from the drawing manager.
        normalized_img_q = self.drawing_manager.get_normalized_image()
        # Convert the QImage to a 0-1 NumPy array.
        numpy_array_0_1 = self.drawing_manager.qimage_to_numpy_array(normalized_img_q)
        # Convert the 0-1 array to a bipolar -1 to 1 array.
        bipolar_array = self._convert_to_bipolar(numpy_array_0_1)

        # Add the bipolar pattern and its target vector to the training lists.
        self.training_patterns.append(bipolar_array)
        self.training_targets.append(target_vector)

        # Provide user feedback.
        QMessageBox.information(self, "Training Data Added",
                                f"Added training pattern. Total patterns: {len(self.training_patterns)}")
        # Clear the canvas to prepare for the next drawing.
        self.drawing_manager.clear_canvas()

        # Retrain the network immediately after adding new data.
        # For a production system, retraining might be batched or done less frequently.
        self._retrain_network()

    def _retrain_network(self):
        """
        Retrains the Hebbian network using all currently collected training patterns
        and their corresponding target vectors. This function is called every time
        a new training pattern is added.
        """
        if not self.training_patterns:
            print("No training patterns available to retrain.")
            return

        print("\n--- Retraining Network ---")
        # Convert the list of patterns and targets into NumPy arrays for efficient training.
        patterns_np = np.array(self.training_patterns)
        targets_np = np.array(self.training_targets)
        # Call the train method of the Hebbian network.
        self.hebbian_net.train(patterns_np, targets_np)
        print("--- Retraining Complete ---\n")

    def _load_and_train_network(self):
        """
        Loads initial dummy training patterns for 'هـ', 'ج', and 'Other/Noise'
        and uses them to perform an initial training of the network.
        These dummy patterns are very basic and serve only as a starting point.
        For accurate recognition, the user must draw and train many real examples
        using the GUI.
        """
        print("Loading initial dummy training patterns...")

        # --- Dummy 'هـ' pattern (a simple ring shape for demonstration) ---
        heh_pattern_0_1 = np.zeros(self.input_neuron_count)
        center_x, center_y = 14, 14
        radius_outer = 8
        radius_inner = 4
        for y in range(self.ann_input_res[1]):
            for x in range(self.ann_input_res[0]):
                dist_sq = (x - center_x)**2 + (y - center_y)**2
                if radius_inner**2 < dist_sq < radius_outer**2:
                    heh_pattern_0_1[y * self.ann_input_res[0] + x] = 1.0 # Black pixel

        # --- Dummy 'ج' pattern (a simple 'L' shape for demonstration) ---
        jeem_pattern_0_1 = np.zeros(self.input_neuron_count)
        # Draw a simple 'L' shape for 'ج'
        for i in range(5, 20): # Vertical part
            jeem_pattern_0_1[i * self.ann_input_res[0] + 10] = 1.0
        for i in range(10, 20): # Horizontal part
            jeem_pattern_0_1[19 * self.ann_input_res[0] + i] = 1.0

        # --- Dummy 'Other' pattern (a single dot for demonstration) ---
        other_pattern_0_1 = np.zeros(self.input_neuron_count)
        other_pattern_0_1[5 * self.ann_input_res[0] + 5] = 1.0 # A dot at top-left

        # Convert these dummy patterns to bipolar format and add to the training data.
        self.training_patterns.append(self._convert_to_bipolar(heh_pattern_0_1))
        self.training_targets.append(self.class_map['heh'])

        self.training_patterns.append(self._convert_to_bipolar(jeem_pattern_0_1))
        self.training_targets.append(self.class_map['jeem'])

        self.training_patterns.append(self._convert_to_bipolar(other_pattern_0_1))
        self.training_targets.append(self.class_map['other'])

        # Perform the initial training of the network with these dummy patterns.
        self._retrain_network()

    def _process_and_recognize(self):
        """
        Captures the current drawing, processes it into a bipolar input pattern,
        feeds it to the trained Hebbian ANN for prediction, and displays the result.
        """
        # Get the normalized QImage from the drawing manager.
        normalized_img_q = self.drawing_manager.get_normalized_image()

        # Update the QLabel to show the user what the normalized input looks like.
        self.normalized_display_label.setPixmap(
            QPixmap.fromImage(normalized_img_q.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        )

        # Convert the QImage to a 0-1 NumPy array and then to bipolar.
        numpy_array_0_1 = self.drawing_manager.qimage_to_numpy_array(normalized_img_q)
        bipolar_array = self._convert_to_bipolar(numpy_array_0_1)

        # Get the activation levels for each output neuron from the Hebbian network.
        activations = self.hebbian_net.predict(bipolar_array)
        print(f"Activations: {activations}") # Print activations for debugging.

        # Find the index of the output neuron with the highest activation.
        predicted_class_idx = np.argmax(activations)
        max_activation_value = activations[predicted_class_idx]

        recognized_char = "Unrecognized"
        # Determine the recognized character based on the highest activation and a threshold.
        # A low maximum activation suggests the drawing doesn't strongly match any learned pattern.
        # The threshold (0.1 here) can be tuned.
        if max_activation_value < 0.1:
             recognized_char = "Other/Unrecognized"
             self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: orange;")
        else:
            # Get the class name from the mapping.
            recognized_char = self.class_names.get(predicted_class_idx, "Unknown")
            # Set text color based on the recognized character for visual feedback.
            if recognized_char == 'هـ':
                self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: green;")
            elif recognized_char == 'ج':
                self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: purple;")
            else: # Fallback, though should be covered by the threshold.
                self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: red;")

        # Update the result label with the recognized character and its "confidence" (max activation).
        self.result_label.setText(f"Recognition Result: {recognized_char} (Confidence: {max_activation_value:.2f})")

        # Save the input image for debugging purposes.
        normalized_img_q.save("last_recognized_input.png")
        print(f"Recognized image saved to last_recognized_input.png (predicted: {recognized_char})")

        # Clear the canvas after recognition, ready for the next drawing.
        self.drawing_manager.clear_canvas()


if __name__ == "__main__":
    # Create the QApplication instance. This is required for any PyQt application.
    app = QApplication(sys.argv)
    # Create an instance of our main application window.
    recognizer_app = RecognitionApp()
    # Show the application window.
    recognizer_app.show()
    # Start the PyQt event loop. This keeps the application running until it's closed.
    sys.exit(app.exec_())
