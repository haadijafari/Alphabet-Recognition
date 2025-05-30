import numpy as np
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap, QTabletEvent
from PyQt5.QtCore import Qt, QPoint, QRect, QSize

class DrawingInputManager(QWidget):
    """
    Manages drawing input from an XP-Pen Deco V2 (or mouse) on a graphical canvas.
    Handles pen movements, pressure sensitivity, and provides methods to clear
    the canvas and extract a normalized image of the drawn content.
    This version directly paints onto a QWidget for simpler visual updates.
    """
    def __init__(self, parent=None, canvas_size=(300, 300), normalized_size=(28, 28)):
        """
        Initializes the drawing canvas.

        Args:
            parent (QWidget): The parent widget for this canvas.
            canvas_size (tuple): The desired initial size (width, height) of the drawing area.
            normalized_size (tuple): The target size (width, height) for the
                                     normalized output image, e.g., (28, 28) for MNIST-like input.
        """
        super().__init__(parent)
        self.setFixedSize(*canvas_size) # Set a fixed size for the drawing area

        # Initialize an internal QImage to draw on. Grayscale8 is efficient for handwriting.
        # It's initially sized to the widget's fixed size.
        self.image = QImage(self.size(), QImage.Format_Grayscale8)
        self.image.fill(Qt.white) # Start with a white canvas

        self.last_point = QPoint() # Stores the last point drawn to connect lines
        self.drawing_active = False # Flag to indicate if the pen/mouse is currently down

        # Configure the pen for drawing. Black color, round caps and joins for smooth strokes.
        # Initial width will be adjusted by pressure.
        self.pen = QPen(Qt.black, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

        # Enable mouse tracking to get events even when buttons aren't pressed (useful for hover, though not used here)
        self.setMouseTracking(True)
        # Enable antialiasing for smoother lines
        self.setAttribute(Qt.WA_StaticContents) # Optimize for static content painting
        # Set cursor to cross-hair when over the canvas
        self.setCursor(Qt.CrossCursor)

        # Store the target size for the normalized output image
        self.normalized_output_size = QSize(*normalized_size)

    def paintEvent(self, event):
        """
        Overrides the paint event to draw the internal QImage onto the widget.
        This is how the drawn content becomes visible.
        """
        painter = QPainter(self) # Create a painter for this widget
        painter.drawImage(0, 0, self.image) # Draw the internal QImage at (0,0)

    def tabletEvent(self, event):
        """
        Handles specific events from a graphics tablet (like XP-Pen).
        This is crucial for capturing pressure and other tablet-specific data.
        """
        # Use event.pointerType() to identify the specific tool being used (pen, eraser, cursor)
        # Qt.PointerType.Pen is the correct enum member for a pen tool.
        if event.pointerType() == QTabletEvent.PointerType.Pen: # <-- NEW CORRECTED LINE
            if event.type() == event.TabletPress:
                self.drawing_active = True
                self.last_point = event.pos() # Record the starting point of the stroke
            elif event.type() == event.TabletMove and self.drawing_active:
                # Draw a line from the last point to the current point on self.image
                painter = QPainter(self.image)
                # Adjust pen width based on pen pressure. Max(1, ...) ensures minimum width.
                # You might need to tune the '10' multiplier for your preference.
                # event.pressure() ranges from 0.0 to 1.0
                self.pen.setWidthF(max(1, event.pressure() * 10))
                painter.setPen(self.pen)
                painter.drawLine(self.last_point, event.pos())
                self.last_point = event.pos() # Update last point for the next segment
                painter.end() # End painter on the QImage
                self.update() # Request a repaint of the widget to show the changes
            elif event.type() == event.TabletRelease:
                self.drawing_active = False # Stop drawing when pen is lifted

            event.accept() # Essential to tell Qt that this event has been handled.
                           # Otherwise, it might be processed as a standard mouse event too.
        else:
            # If it's not a pen (e.g., eraser, or a different deviceType if it exists), ignore it.
            event.ignore()

    def mousePressEvent(self, event):
        """
        Handles mouse press events as a fallback or for testing without a tablet.
        """
        if event.button() == Qt.LeftButton:
            self.drawing_active = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        """
        Handles mouse movement events (for drawing with the mouse).
        """
        if event.buttons() & Qt.LeftButton and self.drawing_active:
            painter = QPainter(self.image)
            self.pen.setWidthF(2) # Fixed width for mouse, as no pressure data is available
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            painter.end() # End painter on the QImage
            self.update() # Request a repaint of the widget to show the changes

    def mouseReleaseEvent(self, event):
        """
        Handles mouse release events.
        """
        if event.button() == Qt.LeftButton:
            self.drawing_active = False

    def clear_canvas(self):
        """
        Clears all drawn content from the canvas by filling it with white.
        """
        self.image.fill(Qt.white)
        self.update() # Request repaint to show the cleared canvas

    def get_normalized_image(self):
        """
        Processes the drawn content to extract a normalized, centered, grayscale image.
        This image is suitable for input to a neural network.

        Returns:
            QImage: A QImage object of the drawn character, normalized to
                    `self.normalized_output_size`, with white background and black foreground.
                    Returns an empty white image if no drawing is detected.
        """
        # 1. Find the bounding box of the non-white pixels (the drawn character)
        min_x, max_x = self.image.width(), 0
        min_y, max_y = self.image.height(), 0
        found_drawing = False

        # Iterate through pixels to find the extent of the drawing
        for y in range(self.image.height()):
            for x in range(self.image.width()):
                # Check for non-white (darker) pixels. Using 250 as threshold to catch faint lines.
                if self.image.pixelColor(x, y).red() < 250:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    found_drawing = True

        if not found_drawing:
            # If no drawing is detected, return an empty (white) normalized image
            normalized_img = QImage(self.normalized_output_size, QImage.Format_Grayscale8)
            normalized_img.fill(Qt.white)
            return normalized_img

        # Add a small padding around the detected character for better context
        padding = 5 # Adjust this value if needed
        min_x = max(0, min_x - padding)
        max_x = min(self.image.width() - 1, max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(self.image.height() - 1, max_y + padding)

        # Ensure the bounding box is valid (at least 1 pixel in each dimension)
        if max_x <= min_x: max_x = min_x + 1
        if max_y <= min_y: max_y = min_y + 1


        # 2. Crop the internal image to the determined bounding box
        cropped_rect = QRect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        cropped_image = self.image.copy(cropped_rect)

        # 3. Scale the cropped image to fit within the target normalized size,
        #    maintaining its aspect ratio.
        scaled_img = cropped_image.scaled(self.normalized_output_size.width(),
                                          self.normalized_output_size.height(),
                                          Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 4. Create a new canvas of the target normalized size and center the scaled image on it.
        normalized_img = QImage(self.normalized_output_size, QImage.Format_Grayscale8)
        normalized_img.fill(Qt.white) # Fill with white background

        painter = QPainter(normalized_img)
        # Calculate offsets to center the scaled image
        x_offset = (self.normalized_output_size.width() - scaled_img.width()) // 2
        y_offset = (self.normalized_output_size.height() - scaled_img.height()) // 2
        painter.drawImage(x_offset, y_offset, scaled_img)
        painter.end()

        return normalized_img

    def qimage_to_numpy_array(self, qimage):
        """
        Converts a QImage (expected to be grayscale) into a flattened NumPy array.
        The pixel values are normalized to 0 (white) to 1 (black) as typically expected
        for neural network input where 1 represents the 'ink'.

        Args:
            qimage (QImage): The QImage to convert.

        Returns:
            np.ndarray: A 1D NumPy array (flattened) of pixel values,
                        where 0 is white and 1 is black.
        """
        if qimage.format() != QImage.Format_Grayscale8:
            # Convert to grayscale if it's not already, for consistent processing
            qimage = qimage.convertToFormat(QImage.Format_Grayscale8)

        width = qimage.width()
        height = qimage.height()

        ptr = qimage.constBits()
        ptr.setsize(height * width)
        img_array = np.frombuffer(ptr, np.uint8).reshape((height, width))

        # Normalize to [0, 1], invert so black = 1, white = 0
        normalized_array = 1.0 - img_array.astype(np.float32) / 255.0

        return normalized_array.flatten()

# --- Example Usage (Main application window to test DrawingInputManager) ---

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("XP-Pen Input Manager Test (Paint-like)")
    window.setGeometry(100, 100, 400, 550) # x, y, width, height

    layout = QVBoxLayout()

    canvas_size = (300, 300)
    ann_input_res = (28, 28) # Target resolution for your ANN

    # Create an instance of our DrawingInputManager
    drawing_manager = DrawingInputManager(parent=window,
                                         canvas_size=canvas_size,
                                         normalized_size=ann_input_res)
    layout.addWidget(drawing_manager)

    # Add buttons for clearing and processing
    clear_button = QPushButton("Clear Drawing")
    clear_button.clicked.connect(drawing_manager.clear_canvas)
    layout.addWidget(clear_button)

    process_button = QPushButton("Process and Show Normalized Image")
    # This lambda function allows us to call a function with arguments
    # and display the result.
    def process_and_show():
        # Get the normalized QImage
        normalized_img_q = drawing_manager.get_normalized_image()
        # Save it to a file for verification during development
        normalized_img_q.save("output_normalized_input.png")
        print(f"Normalized image saved to output_normalized_input.png (size: {normalized_img_q.width()}x{normalized_img_q.height()})")

        # Convert the QImage to a NumPy array for your ANN
        numpy_array = drawing_manager.qimage_to_numpy_array(normalized_img_q)
        print(f"NumPy array shape: {numpy_array.shape}")
        # print(f"NumPy array (first 20 pixels): {numpy_array[:20]}") # Uncomment to see raw pixel data

        # Display the normalized image in a new temporary window for visual check
        display_window = QWidget()
        display_window.setWindowTitle("Normalized Input for ANN")
        display_layout = QVBoxLayout()
        display_label = QLabel()
        # Scale the displayed image up so it's clearly visible
        display_label.setPixmap(QPixmap.fromImage(normalized_img_q.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        display_layout.addWidget(display_label)
        display_window.setLayout(display_layout)
        display_window.show()

    process_button.clicked.connect(process_and_show)
    layout.addWidget(process_button)

    window.setLayout(layout)
    window.show()
    sys.exit(app.exec_())