"""
Single-layer learning algorithms for character recognition.

Every algorithm uses the same network shape: a weight matrix of shape
(input_size, output_size) plus an optional bias per output neuron. Only the
learning rule (`_fit`) and how outputs are scored and thresholded differ, so the
app can switch between them and retrain on the same patterns.

Conventions shared by all algorithms:
    - Input patterns are unipolar, in [0, 1] (0 = background, 1 = ink).
    - Target vectors are bipolar (-1 or 1), one element per output neuron.
    - A pattern whose highest score is below `decision_threshold` belongs to
      none of the output classes ("Other/Unrecognized").
"""

import numpy as np


class LinearNetwork:
    """
    Base class: a single layer of output neurons with weights and biases.
    Subclasses implement `_fit` (the learning rule) and may override `scores`
    and `decision_threshold`.
    """
    display_name = "Linear network"
    # Minimum score for the best output neuron to count as a recognized class.
    decision_threshold = 0.0

    def __init__(self, input_size: int, output_size: int, learning_rate: float = 1.0):
        """
        Args:
            input_size (int): The number of input neurons, e.g. 28*28 = 784 for a 28x28 image.
            output_size (int): The number of output neurons, one per class to recognize.
            learning_rate (float): Positive step size for each weight update.
        Raises:
            ValueError: If learning_rate is not a positive number.
        """
        if not learning_rate > 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        # Each column holds the weight vector of one output neuron.
        self.weights = np.zeros((input_size, output_size))
        self.bias = np.zeros(output_size)
        # A short description of the last training run, for display.
        self.training_summary = "Not trained yet."

    def train(self, patterns: np.ndarray, targets: np.ndarray):
        """
        Trains the network from scratch on all given patterns.

        Args:
            patterns (np.ndarray): Shape (num_samples, input_size), values in [0, 1].
            targets (np.ndarray): Shape (num_samples, output_size), bipolar values (-1 or 1).
        Raises:
            ValueError: If the dimensions do not match the network's configuration.
        """
        patterns = np.asarray(patterns, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if patterns.ndim != 2 or patterns.shape[1] != self.input_size:
            raise ValueError(f"Patterns must have shape (n, {self.input_size}), got {patterns.shape}")
        if targets.ndim != 2 or targets.shape[1] != self.output_size:
            raise ValueError(f"Targets must have shape (n, {self.output_size}), got {targets.shape}")
        if patterns.shape[0] != targets.shape[0]:
            raise ValueError("Number of patterns and targets must be the same.")

        self.weights = np.zeros((self.input_size, self.output_size))
        self.bias = np.zeros(self.output_size)
        print(f"Training {self.display_name} on {patterns.shape[0]} patterns...")
        self._fit(patterns, targets)
        print(f"Training complete: {self.training_summary}")

    def _fit(self, patterns: np.ndarray, targets: np.ndarray):
        raise NotImplementedError

    def predict(self, input_pattern: np.ndarray) -> np.ndarray:
        """
        Returns the raw activation (weighted sum plus bias) of each output neuron.

        Raises:
            ValueError: If the input pattern's size does not match the network's input size.
        """
        if len(input_pattern) != self.input_size:
            raise ValueError(f"Input pattern size {len(input_pattern)} does not match network input size {self.input_size}")
        return np.dot(input_pattern, self.weights) + self.bias

    def scores(self, input_pattern: np.ndarray) -> np.ndarray:
        """
        Returns one score per output neuron, compared against `decision_threshold`.
        By default this is the raw activation.
        """
        return self.predict(input_pattern)

    def classify(self, input_pattern: np.ndarray):
        """
        Returns (index of the winning output neuron, its score). The index is None
        if even the best score is below `decision_threshold` (no class matches).
        """
        scores = self.scores(input_pattern)
        best = int(np.argmax(scores))
        if scores[best] < self.decision_threshold:
            return None, float(scores[best])
        return best, float(scores[best])


class HebbianANN(LinearNetwork):
    """
    Hebbian learning: "neurons that fire together, wire together".
    Each pattern adds learning_rate * x_i * y_j to weight w_ij, whether or not the
    network already classifies it correctly. Uses no bias.
    """
    display_name = "Hebbian"
    # Applied to the normalized score from `scores` (cosine similarity, in [-1, 1]).
    decision_threshold = 0.1

    def _fit(self, patterns, targets):
        # Summing learning_rate * x * y over every pattern is the matrix product
        # patterns^T @ targets, giving shape (input_size, output_size).
        self.weights = self.learning_rate * (patterns.T @ targets)
        self.training_summary = f"one pass over {patterns.shape[0]} patterns"

    def similarity(self, input_pattern: np.ndarray) -> np.ndarray:
        """
        Returns the activations normalized to [-1, 1] (cosine similarity between the
        input and each output neuron's weight vector).

        Raw Hebbian activations grow with the number of inputs, the learning rate and
        the number of training patterns, so a fixed threshold on them has no stable
        meaning. The normalized score does not depend on any of those.

        Returns:
            np.ndarray: One score per output neuron. 0 for an all-zero input or a
                        neuron with all-zero weights.
        """
        net_inputs = self.predict(input_pattern)
        norms = np.linalg.norm(input_pattern) * np.linalg.norm(self.weights, axis=0)
        return np.divide(net_inputs, norms, out=np.zeros_like(net_inputs, dtype=float), where=norms > 0)

    def scores(self, input_pattern):
        return self.similarity(input_pattern)


class PerceptronANN(LinearNetwork):
    """
    Perceptron learning rule: weights change only when a neuron's output is wrong.
    For each pattern and neuron j with output y_j = sign(w_j . x + b_j):
        w_j += learning_rate * (t_j - y_j) / 2 * x
    which is learning_rate * t_j * x on a mistake and nothing otherwise.
    Repeats over the data until an epoch has no mistakes or `max_epochs` is reached.
    A neuron fires (claims its class) when its activation is above 0.
    """
    display_name = "Perceptron"
    decision_threshold = 0.0

    def __init__(self, input_size, output_size, learning_rate=1.0, max_epochs=100, seed=0):
        super().__init__(input_size, output_size, learning_rate)
        self.max_epochs = max_epochs
        self.seed = seed

    def _fit(self, patterns, targets):
        rng = np.random.default_rng(self.seed)  # Fixed seed: same data gives the same weights.
        for epoch in range(1, self.max_epochs + 1):
            mistakes = 0
            for i in rng.permutation(patterns.shape[0]):
                x, t = patterns[i], targets[i]
                y = np.where(x @ self.weights + self.bias > 0, 1.0, -1.0)
                error = (t - y) / 2  # -1, 0 or 1 per neuron.
                if error.any():
                    mistakes += 1
                    self.weights += self.learning_rate * np.outer(x, error)
                    self.bias += self.learning_rate * error
            if mistakes == 0:
                self.training_summary = f"converged after {epoch} epochs"
                return
        self.training_summary = (f"did not converge in {self.max_epochs} epochs "
                                 f"({mistakes} patterns still misclassified in the last epoch)")


class AdalineANN(LinearNetwork):
    """
    Delta rule (Adaline / Widrow-Hoff LMS): minimizes the squared error between the
    raw activation and the target, so it keeps adjusting even correctly classified
    patterns toward exactly +1 / -1.
        w_j += learning_rate * (t_j - net_j) * x / (|x|^2 + 1)
    Dividing by |x|^2 + 1 (normalized LMS) keeps the step size stable no matter how
    much ink a drawing has; learning_rate should be between 0 and 2.
    A neuron fires (claims its class) when its activation is above 0.
    """
    display_name = "Delta rule (Adaline)"
    decision_threshold = 0.0

    def __init__(self, input_size, output_size, learning_rate=0.1, epochs=50, seed=0):
        if not 0 < learning_rate < 2:
            raise ValueError(f"learning_rate must be between 0 and 2 for normalized LMS, got {learning_rate}")
        super().__init__(input_size, output_size, learning_rate)
        self.epochs = epochs
        self.seed = seed

    def _fit(self, patterns, targets):
        rng = np.random.default_rng(self.seed)  # Fixed seed: same data gives the same weights.
        for _ in range(self.epochs):
            for i in rng.permutation(patterns.shape[0]):
                x, t = patterns[i], targets[i]
                error = t - (x @ self.weights + self.bias)
                step = self.learning_rate / (x @ x + 1)
                self.weights += step * np.outer(x, error)
                self.bias += step * error
        mse = np.mean((targets - (patterns @ self.weights + self.bias)) ** 2)
        self.training_summary = f"{self.epochs} epochs, mean squared error {mse:.3f}"


# Algorithms the app can switch between, by command-line key.
ALGORITHMS = {
    'hebbian': HebbianANN,
    'perceptron': PerceptronANN,
    'adaline': AdalineANN,
}
