import argparse
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QHBoxLayout, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize

# Import the DrawingInputManager class from your local file.
# Make sure 'drawing_input_manager.py' is in the same directory as this script.
from drawing_input_manager import DrawingInputManager
# HebbianANN is re-exported here so existing imports from this module keep working.
from learning_algorithms import ALGORITHMS, HebbianANN

class RecognitionApp(QWidget):
    """
    A PyQt5 application that integrates the DrawingInputManager with a single-layer
    network (Hebbian, Perceptron or Delta rule, switchable at runtime) to allow users
    to draw characters, train the network with examples, and then recognize new drawings.
    """
    DEFAULT_ALGORITHM = 'hebbian'

    # Training patterns are stored as <data_dir>/<class key>/<timestamp>.png.
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "training_data"

    def __init__(self, data_dir=None, algorithm=DEFAULT_ALGORITHM):
        """
        Initializes the recognition application, sets up the UI,
        and prepares the Hebbian ANN for training and prediction.

        Args:
            data_dir (str | Path | None): Directory where training patterns are saved
                                          and loaded from. Defaults to DEFAULT_DATA_DIR.
            algorithm (str): Key in learning_algorithms.ALGORITHMS of the learning
                             algorithm to start with.
        Raises:
            ValueError: If algorithm is not a known key.
        """
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {', '.join(ALGORITHMS)}")
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir is not None else self.DEFAULT_DATA_DIR
        self.setWindowTitle("Persian Character Recognizer (هـ, ج)")
        self.algorithm_key = algorithm
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
        # Human-readable labels for each training class key (used in messages).
        self.class_labels = {'heh': 'هـ', 'jeem': 'ج', 'other': 'Other'}

        # Initialize the network for the selected learning algorithm.
        self.network = self._create_network(self.algorithm_key)
        # The user's training patterns, in the order they were added (oldest first).
        # Each entry is a dict with 'pattern' (input vector), 'class_key' and 'path'
        # (the saved PNG). Built-in starter patterns are not stored here.
        self.training_entries = []

        # Set up the graphical user interface.
        self._init_ui()
        # Load the saved training patterns (or the built-in starter patterns if
        # there are none) and train the network.
        self._load_and_train_network()

    def _init_ui(self):
        """
        Sets up the layout and widgets for the application window.
        Includes the drawing canvas, clear button, training buttons,
        recognition button, and result display.
        """
        main_layout = QVBoxLayout() # Main vertical layout for the entire window.

        # Drop-down to switch the learning algorithm. Switching retrains the new
        # network on the same training patterns.
        algorithm_layout = QHBoxLayout()
        algorithm_layout.addWidget(QLabel("Learning algorithm:"))
        self.algorithm_combo = QComboBox()
        for key, network_class in ALGORITHMS.items():
            self.algorithm_combo.addItem(network_class.display_name, key)
        self.algorithm_combo.setCurrentIndex(self.algorithm_combo.findData(self.algorithm_key))
        # Connected after setting the initial index so it does not fire during setup.
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        algorithm_layout.addWidget(self.algorithm_combo, 1)
        main_layout.addLayout(algorithm_layout)

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
        # Connect to a lambda function to pass the class key for 'هـ'.
        train_heh_button.clicked.connect(lambda: self._add_training_pattern('heh'))
        training_buttons_layout.addWidget(train_heh_button)

        # Button to train the current drawing as 'ج'.
        train_jeem_button = QPushButton("Train as 'ج'")
        train_jeem_button.clicked.connect(lambda: self._add_training_pattern('jeem'))
        training_buttons_layout.addWidget(train_jeem_button)

        # Button to train the current drawing as 'Other/Noise' (neither 'هـ' nor 'ج').
        train_other_button = QPushButton("Train as 'Other/Noise'")
        train_other_button.clicked.connect(lambda: self._add_training_pattern('other'))
        training_buttons_layout.addWidget(train_other_button)

        # Add the horizontal layout of training buttons to the main layout.
        main_layout.addLayout(training_buttons_layout)

        # Button to remove the most recently added training pattern (and its saved file).
        self.undo_button = QPushButton("Undo Last Training Pattern")
        self.undo_button.clicked.connect(self._undo_last_pattern)
        main_layout.addWidget(self.undo_button)

        # Label showing how many saved patterns each class has.
        self.training_status_label = QLabel()
        self.training_status_label.setAlignment(Qt.AlignCenter)
        self.training_status_label.setWordWrap(True)
        main_layout.addWidget(self.training_status_label)

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

    def _current_input_pattern(self):
        """
        Returns the current drawing as a normalized QImage and as the network's
        input vector.

        Inputs are unipolar: 0 for white background, 1 for black ink. Background
        pixels (about 90% of the image) therefore contribute nothing to the Hebbian
        weights or the dot product, so the network compares stroke shapes instead
        of being dominated by the shared white background, as it is with a bipolar
        (-1 for white) encoding. Targets stay bipolar, so a -1 target still pushes
        the weights of that pattern's ink pixels down.
        """
        normalized_img_q = self.drawing_manager.get_normalized_image()
        pattern = self.drawing_manager.qimage_to_numpy_array(normalized_img_q)
        return normalized_img_q, pattern

    def _warn_empty_canvas(self, action: str) -> bool:
        """
        Shows a warning and returns True if the canvas is blank.
        """
        if self.drawing_manager.has_drawing():
            return False
        QMessageBox.warning(self, "Empty Canvas", f"Draw a character before you {action}.")
        return True

    def _add_training_pattern(self, class_key: str):
        """
        Captures the current drawing from the `DrawingInputManager`, saves it as a
        PNG under `data_dir/<class_key>/`, adds it to the training dataset, clears
        the canvas and retrains the network. Does nothing (apart from a warning)
        if the canvas is blank or the file cannot be saved.

        Args:
            class_key (str): One of the keys of `class_map` ('heh', 'jeem' or 'other').
        """
        # A blank canvas would add an empty pattern with this label to the training set.
        if self._warn_empty_canvas("train"):
            return

        normalized_img_q, pattern = self._current_input_pattern()

        # Save the normalized image so the pattern is reloaded on the next launch.
        # Grayscale PNG is lossless, so the reloaded input vector is identical.
        class_dir = self.data_dir / class_key
        try:
            class_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "Could Not Save Pattern", f"Could not create {class_dir}:\n{e}")
            return
        # Timestamped names sort in the order patterns were added (used for undo).
        path = class_dir / f"{datetime.now():%Y%m%d-%H%M%S-%f}.png"
        if not normalized_img_q.save(str(path)):
            QMessageBox.warning(self, "Could Not Save Pattern", f"Could not write {path}.")
            return

        self.training_entries.append({'pattern': pattern, 'class_key': class_key, 'path': path})

        # Provide user feedback.
        QMessageBox.information(self, "Training Data Added",
                                f"Added a '{self.class_labels[class_key]}' pattern. "
                                f"Total saved patterns: {len(self.training_entries)}")
        # Clear the canvas to prepare for the next drawing.
        self.drawing_manager.clear_canvas()

        # Retrain the network immediately after adding new data.
        self._retrain_network()

    def _undo_last_pattern(self):
        """
        Removes the most recently added training pattern, deletes its saved file
        and retrains the network. Works across launches, since saved patterns are
        loaded in the order they were added.
        """
        if not self.training_entries:
            QMessageBox.information(self, "Nothing to Undo", "There are no saved training patterns.")
            return

        entry = self.training_entries[-1]
        try:
            entry['path'].unlink(missing_ok=True)
        except OSError as e:
            # Keep the pattern: otherwise it would silently come back on the next launch.
            QMessageBox.warning(self, "Could Not Undo", f"Could not delete {entry['path']}:\n{e}")
            return
        self.training_entries.pop()

        QMessageBox.information(self, "Pattern Removed",
                                f"Removed the last '{self.class_labels[entry['class_key']]}' pattern. "
                                f"Total saved patterns: {len(self.training_entries)}")
        self._retrain_network()

    def _load_saved_patterns(self):
        """
        Loads the training patterns saved under `data_dir`, oldest first.
        Files that cannot be read or have the wrong size are skipped with a message.
        """
        files = []
        for class_key in self.class_map:
            class_dir = self.data_dir / class_key
            if class_dir.is_dir():
                files.extend((path, class_key) for path in class_dir.glob("*.png"))
        # Filenames are timestamps, so sorting by name restores the order they were added.
        files.sort(key=lambda item: item[0].name)

        expected_size = QSize(*self.ann_input_res)
        for path, class_key in files:
            image = QImage(str(path))
            if image.isNull() or image.size() != expected_size:
                print(f"Skipping {path}: not a readable "
                      f"{self.ann_input_res[0]}x{self.ann_input_res[1]} image.")
                continue
            pattern = self.drawing_manager.qimage_to_numpy_array(image)
            self.training_entries.append({'pattern': pattern, 'class_key': class_key, 'path': path})
        print(f"Loaded {len(self.training_entries)} saved training patterns from {self.data_dir}")

    def _retrain_network(self):
        """
        Retrains the current network from scratch. Uses the user's saved patterns,
        or the built-in starter patterns while there are none, and updates the
        training status label.
        """
        print("\n--- Retraining Network ---")
        if self.training_entries:
            patterns = [entry['pattern'] for entry in self.training_entries]
            targets = [self.class_map[entry['class_key']] for entry in self.training_entries]
        else:
            print("No saved training patterns; using the built-in starter patterns.")
            patterns, targets = self._starter_patterns()
        self.network.train(np.array(patterns), np.array(targets))
        print("--- Retraining Complete ---\n")
        self._update_training_status()

    def _create_network(self, algorithm_key: str):
        """
        Creates an untrained network for the given algorithm key.
        """
        return ALGORITHMS[algorithm_key](input_size=self.input_neuron_count, output_size=self.num_classes)

    def _on_algorithm_changed(self, index: int):
        """
        Switches to the algorithm selected in the drop-down and retrains on the
        same training patterns.
        """
        self.algorithm_key = self.algorithm_combo.itemData(index)
        self.network = self._create_network(self.algorithm_key)
        self._retrain_network()
        self.result_label.setText(f"Switched to {self.network.display_name}. Draw and click 'Recognize'")
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: blue;")

    def _class_key_for(self, class_idx):
        """
        Maps a classify() result (output neuron index, or None) to a class key.
        """
        if class_idx is None:
            return 'other'
        target = -np.ones(self.num_classes)
        target[class_idx] = 1
        for key, vector in self.class_map.items():
            if np.array_equal(vector, target):
                return key
        return 'other'

    def _update_training_status(self):
        """
        Shows the number of saved patterns for each class, the current algorithm's
        training result, and how many saved patterns it classifies correctly.
        """
        counts = {key: 0 for key in self.class_map}
        for entry in self.training_entries:
            counts[entry['class_key']] += 1
        summary = ", ".join(f"{self.class_labels[key]}: {count}" for key, count in counts.items())
        text = f"Saved patterns: {summary}"
        text += f"\n{self.network.display_name}: {self.network.training_summary}"
        if self.training_entries:
            # Accuracy on the training set itself: shows how well each algorithm can
            # fit the same data (not how well it generalizes to new drawings).
            correct = sum(self._class_key_for(self.network.classify(entry['pattern'])[0]) == entry['class_key']
                          for entry in self.training_entries)
            total = len(self.training_entries)
            text += f"\nTraining accuracy: {correct}/{total} ({100 * correct / total:.0f}%)"
        else:
            text += "\n(using built-in starter patterns until you train your own)"
        self.training_status_label.setText(text)
        self.undo_button.setEnabled(bool(self.training_entries))

    def _load_and_train_network(self):
        """
        Loads the saved training patterns and performs the initial training.
        """
        self._load_saved_patterns()
        self._retrain_network()

    def _starter_patterns(self):
        """
        Returns very basic built-in patterns for 'هـ', 'ج', and 'Other/Noise', used
        only while no training patterns have been saved. For accurate recognition,
        the user must draw and train many real examples using the GUI.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: The patterns and their target vectors.
        """
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

        # These are already unipolar (0 = background, 1 = ink).
        patterns = [heh_pattern_0_1, jeem_pattern_0_1, other_pattern_0_1]
        targets = [self.class_map['heh'], self.class_map['jeem'], self.class_map['other']]
        return patterns, targets

    def _process_and_recognize(self):
        """
        Captures the current drawing, processes it into an input pattern,
        feeds it to the current network for prediction, and displays the result.
        Does nothing (apart from a warning) if the canvas is blank.
        """
        if self._warn_empty_canvas("recognize"):
            return

        normalized_img_q, pattern = self._current_input_pattern()

        # Update the QLabel to show the user what the normalized input looks like.
        self.normalized_display_label.setPixmap(
            QPixmap.fromImage(normalized_img_q.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        )

        # Each algorithm scores the output neurons on its own scale and decides
        # whether the best one is strong enough (otherwise: no class matches).
        predicted_class_idx, max_activation_value = self.network.classify(pattern)
        print(f"Activations: {self.network.predict(pattern)}, scores: {self.network.scores(pattern)}") # For debugging.

        if predicted_class_idx is None:
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

        # Update the result label with the recognized character and its score. The
        # score's scale depends on the algorithm (cosine for Hebbian, raw activation otherwise).
        self.result_label.setText(f"Recognition Result: {recognized_char} (Score: {max_activation_value:.2f})")

        # Clear the canvas after recognition, ready for the next drawing.
        self.drawing_manager.clear_canvas()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw and recognize Persian characters.")
    parser.add_argument("--algorithm", choices=ALGORITHMS, default=RecognitionApp.DEFAULT_ALGORITHM,
                        help="learning algorithm to start with (can be switched in the window)")
    parser.add_argument("--data-dir", default=None,
                        help=f"directory for saved training patterns (default: {RecognitionApp.DEFAULT_DATA_DIR})")
    # Unrecognized arguments are passed on to Qt (e.g. -platform).
    args, qt_args = parser.parse_known_args()

    # Create the QApplication instance. This is required for any PyQt application.
    app = QApplication(sys.argv[:1] + qt_args)
    # Create an instance of our main application window.
    recognizer_app = RecognitionApp(data_dir=args.data_dir, algorithm=args.algorithm)
    # Show the application window.
    recognizer_app.show()
    # Start the PyQt event loop. This keeps the application running until it's closed.
    sys.exit(app.exec_())
