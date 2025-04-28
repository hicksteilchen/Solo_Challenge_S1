# write ROC function


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Example Input Data
input_array_1 = np.array([1.001, 1.002, 1.003, 42000])  # No spike trials (negative class)
input_array_2 = np.array([1.0021, 1.0028, 1.0029, 1.0027])  # Spike occurred (positive class)

# Prepare Data for Training
X = np.concatenate((input_array_1 , input_array_2))
y = np.array([0] * len(input_array_1) + [1] * len(input_array_2))

# Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


class ROCAnalysis:
    def __init__(self, X_train, y_train):
        """
        Initialize the class with input arrays.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.optimal_threshold = None
        self.y_pred_probs = np.random.poisson(
            lam=3, size=len(self.X_train)
        )  # Poisson-based probabilities

    def compute_confusion_matrix(self, threshold):
        """
        Compute TP, FP, TN, FN based on a given threshold.
        """
        y_pred_class = (self.y_pred_probs >= threshold).astype(int)  # Apply threshold

        TP = np.sum((y_pred_class == 1) & (self.y_train == 1))  # True Positives
        FP = np.sum((y_pred_class == 1) & (self.y_train == 0))  # False Positives
        TN = np.sum((y_pred_class == 0) & (self.y_train == 0))  # True Negatives
        FN = np.sum((y_pred_class == 0) & (self.y_train == 1))  # False Negatives

        return TP, FP, TN, FN

    def TPR(self, TP, FN):
        """Compute True Positive Rate (Recall)."""
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    def FPR(self, FP, TN):
        """Compute False Positive Rate."""
        return FP / (FP + TN) if (FP + TN) > 0 else 0

    def find_optimal_threshold(self):
        """
        Iterate over different threshold values and find the one where the ROC curve slope is closest to 1.
        """
        possible_thresholds = np.arange(
            0, max(self.y_pred_probs) + 1, 1
        )  # Possible thresholds
        tpr_values = []
        fpr_values = []

        for z in possible_thresholds:
            TP, FP, TN, FN = self.compute_confusion_matrix(z)
            tpr_values.append(self.TPR(TP, FN))
            fpr_values.append(self.FPR(FP, TN))

        # Compute slopes of ROC segments (ΔTPR / ΔFPR)
        slopes = np.diff(tpr_values) / (
            np.diff(fpr_values) + 1e-6
        )  # Avoid division by zero

        # Find threshold where slope ≈ 1 (closest to a 45-degree diagonal)
        best_index = np.argmin(np.abs(slopes - 1))  # Index of the best threshold
        self.optimal_threshold = possible_thresholds[best_index]

        print(f"Optimal threshold selected: {self.optimal_threshold}")

    def plot_roc_curve(self):
        """
        Generate and plot the ROC curve.
        """
        fpr, tpr, thresholds = roc_curve(self.y_train, self.y_pred_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(
            fpr,
            tpr,
            marker="o",
            linestyle="-",
            color="blue",
            label=f"ROC curve (AUC = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve with Optimal Threshold Selection")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    def predict(self, value):
        """
        Predict whether a given value comes from class 0 (no spike) or class 1 (spike) using optimal threshold.
        """
        return 1 if value >= self.optimal_threshold else 0  # 1 = Spike, 0 = No Spike

    def predict_batch(self, values):
        """
        Predict for a batch of values using optimal threshold.
        """
        return np.array([self.predict(v) for v in values])


# Initialize ROC Analysis
roc = ROCAnalysis(X_train, y_train)

# Find the best threshold using slope closest to 1
roc.find_optimal_threshold()

# Plot ROC Curve
roc.plot_roc_curve()

# Select random test values from X_test and y_test
test_values = np.random.choice(X_test, size=min(3, len(X_test)), replace=False)
predictions = roc.predict_batch(test_values)

print(f"Test values: {test_values}")
print(f"Predictions: {predictions} (0 = No Spike, 1 = Spike)")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Example Input Data
input_array_1 = [0, 1, 2, 1, 2]  # No spike trials (negative class)
input_array_2 = [4, 5, 6, 5, 7]  # Spike occurred (positive class)

# Prepare Data for Training
X = np.array(input_array_1 + input_array_2)
y = np.array([0] * len(input_array_1) + [1] * len(input_array_2))

# Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


class ROCAnalysis:
    def __init__(self, X_train, y_train):
        """
        Initialize the class with input arrays.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.optimal_threshold = None
        self.y_pred_probs = np.random.poisson(
            lam=3, size=len(self.X_train)
        )  # Poisson-based probabilities

    def compute_confusion_matrix(self, threshold):
        """
        Compute TP, FP, TN, FN based on a given threshold.
        """
        y_pred_class = (self.y_pred_probs >= threshold).astype(int)  # Apply threshold

        TP = np.sum((y_pred_class == 1) & (self.y_train == 1))  # True Positives
        FP = np.sum((y_pred_class == 1) & (self.y_train == 0))  # False Positives
        TN = np.sum((y_pred_class == 0) & (self.y_train == 0))  # True Negatives
        FN = np.sum((y_pred_class == 0) & (self.y_train == 1))  # False Negatives

        return TP, FP, TN, FN

    def TPR(self, TP, FN):
        """Compute True Positive Rate (Recall)."""
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    def FPR(self, FP, TN):
        """Compute False Positive Rate."""
        return FP / (FP + TN) if (FP + TN) > 0 else 0

    def accuracy(self, TP, FP, TN, FN):
        """
        Compute classification accuracy: (TP + TN) / (TP + FP + TN + FN)
        """
        return (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0

    def find_optimal_threshold(self):
        """
        Iterate over different threshold values and find the one where:
          1. The ROC curve slope is closest to 1.
          2. The classification accuracy is maximized.
        """
        possible_thresholds = np.arange(
            0, max(self.y_pred_probs) + 1, 1
        )  # Possible thresholds
        tpr_values = []
        fpr_values = []
        acc_values = []

        for z in possible_thresholds:
            TP, FP, TN, FN = self.compute_confusion_matrix(z)
            tpr_values.append(self.TPR(TP, FN))
            fpr_values.append(self.FPR(FP, TN))
            acc_values.append(self.accuracy(TP, FP, TN, FN))

        # Compute slopes of ROC segments (ΔTPR / ΔFPR)
        slopes = np.diff(tpr_values) / (
            np.diff(fpr_values) + 1e-6
        )  # Avoid division by zero

        # Find threshold where slope ≈ 1 (closest to a 45-degree diagonal)
        slope_best_index = np.argmin(np.abs(slopes - 1))  # Index of the best threshold

        # Find the threshold that maximizes accuracy
        acc_best_index = np.argmax(acc_values)

        # Choose the best threshold based on accuracy and ROC slope
        self.optimal_threshold = possible_thresholds[acc_best_index]

        print(f"Optimal threshold selected: {self.optimal_threshold}")
        print(
            f"Max Accuracy at z={self.optimal_threshold}: {acc_values[acc_best_index]:.2f}"
        )

    def plot_roc_curve(self):
        """
        Generate and plot the ROC curve.
        """
        fpr, tpr, thresholds = roc_curve(self.y_train, self.y_pred_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(
            fpr,
            tpr,
            marker="o",
            linestyle="-",
            color="blue",
            label=f"ROC curve (AUC = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve with Optimal Threshold Selection")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    def predict(self, value):
        """
        Predict whether a given value comes from class 0 (no spike) or class 1 (spike) using optimal threshold.
        """
        return 1 if value >= self.optimal_threshold else 0  # 1 = Spike, 0 = No Spike

    def predict_batch(self, values):
        """
        Predict for a batch of values using optimal threshold.
        """
        return np.array([self.predict(v) for v in values])


# Initialize ROC Analysis
roc = ROCAnalysis(X_train, y_train)

# Find the best threshold using slope closest to 1 and max accuracy
roc.find_optimal_threshold()

# Plot ROC Curve
roc.plot_roc_curve()

# Select random test values from X_test and y_test
test_values = np.random.choice(X_test, size=min(3, len(X_test)), replace=False)
predictions = roc.predict_batch(test_values)

print(f"Test values: {test_values}")
print(f"Predictions: {predictions} (0 = No Spike, 1 = Spike)")

# %%


# Function to generate a Poisson spike train
def generate_poisson_spike_train(rate, T, dt=0.001):
    """
    Generates a Poisson spike train for a given firing rate over time T.

    Args:
        rate (float): Firing rate (Hz)
        T (float): Time window (seconds)
        dt (float): Time step for binning (default: 1 ms)

    Returns:
        np.array: Binned spike count representation
    """
    num_bins = int(T / dt)
    spikes = np.random.rand(num_bins) < rate * dt  # Bernoulli trials
    return spikes.astype(int)  # Convert to 0s and 1s


# Parameters
T = 1.0  # Total duration in seconds
dt = 0.01  # Bin width (10 ms bins)
r1 = 10  # Firing rate of neuron 1 (Hz)
r2 = 30  # Firing rate of neuron 2 (Hz)

# Generate spike trains for two neurons
spike_train_1 = generate_poisson_spike_train(r1, T, dt)
spike_train_2 = generate_poisson_spike_train(r2, T, dt)

# Convert spike trains into binned spike counts
bin_edges = np.arange(0, T, dt)
input_array_1 = np.histogram(bin_edges, bins=len(spike_train_1), weights=spike_train_1)[
    0
]
input_array_2 = np.histogram(bin_edges, bins=len(spike_train_2), weights=spike_train_2)[
    0
]

# Prepare data for ROC analysis
X = np.concatenate([input_array_1, input_array_2])
y = np.concatenate([np.zeros(len(input_array_1)), np.ones(len(input_array_2))])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


class ROCAnalysis:
    def __init__(self, X_train, y_train):
        """
        Initialize the class with input arrays.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.optimal_threshold = None
        self.y_pred_probs = np.random.poisson(
            lam=np.mean(self.X_train), size=len(self.X_train)
        )

    def compute_confusion_matrix(self, threshold):
        """
        Compute TP, FP, TN, FN based on a given threshold.
        """
        y_pred_class = (self.y_pred_probs >= threshold).astype(int)

        TP = np.sum((y_pred_class == 1) & (self.y_train == 1))
        FP = np.sum((y_pred_class == 1) & (self.y_train == 0))
        TN = np.sum((y_pred_class == 0) & (self.y_train == 0))
        FN = np.sum((y_pred_class == 0) & (self.y_train == 1))

        return TP, FP, TN, FN

    def TPR(self, TP, FN):
        """Compute True Positive Rate (Recall)."""
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    def FPR(self, FP, TN):
        """Compute False Positive Rate."""
        return FP / (FP + TN) if (FP + TN) > 0 else 0

    def accuracy(self, TP, FP, TN, FN):
        """
        Compute classification accuracy: (TP + TN) / (TP + FP + TN + FN)
        """
        return (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0

    def find_optimal_threshold(self):
        """
        Iterate over different threshold values and find the one where:
          1. The ROC curve slope is closest to 1.
          2. The classification accuracy is maximized.
        """
        possible_thresholds = np.arange(0, max(self.y_pred_probs) + 1, 1)
        tpr_values = []
        fpr_values = []
        acc_values = []

        for z in possible_thresholds:
            TP, FP, TN, FN = self.compute_confusion_matrix(z)
            tpr_values.append(self.TPR(TP, FN))
            fpr_values.append(self.FPR(FP, TN))
            acc_values.append(self.accuracy(TP, FP, TN, FN))

        # Compute slopes of ROC segments
        slopes = np.diff(tpr_values) / (np.diff(fpr_values) + 1e-6)

        # Find threshold where slope ≈ 1
        slope_best_index = np.argmin(np.abs(slopes - 1))

        # Find the threshold that maximizes accuracy
        acc_best_index = np.argmax(acc_values)

        # Choose the best threshold based on accuracy
        self.optimal_threshold = possible_thresholds[acc_best_index]

        print(f"Optimal threshold selected: {self.optimal_threshold}")
        print(
            f"Max Accuracy at z={self.optimal_threshold}: {acc_values[acc_best_index]:.2f}"
        )

    def plot_roc_curve(self):
        """
        Generate and plot the ROC curve.
        """
        fpr, tpr, thresholds = roc_curve(self.y_train, self.y_pred_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(
            fpr,
            tpr,
            marker="o",
            linestyle="-",
            color="blue",
            label=f"ROC curve (AUC = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve with Poisson Neurons")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    def predict(self, value):
        """
        Predict whether a given value comes from class 0 (low rate) or class 1 (high rate).
        """
        return 1 if value >= self.optimal_threshold else 0

    def predict_batch(self, values):
        """
        Predict for a batch of values using optimal threshold.
        """
        return np.array([self.predict(v) for v in values])


# Initialize and Run Analysis
roc = ROCAnalysis(X_train, y_train)

# Find the best threshold
roc.find_optimal_threshold()

# Plot ROC Curve
roc.plot_roc_curve()

# Predict on random test values
test_values = np.random.choice(X_test, size=min(3, len(X_test)), replace=False)
predictions = roc.predict_batch(test_values)

print(f"Test values: {test_values}")
print(f"Predictions: {predictions} (0 = Neuron 1, 1 = Neuron 2)")

# %%

# Parameters
T = 1.0  # Time interval in seconds
r1_values = np.linspace(1, 50, 50)  # Firing rates for neuron 1 (Hz)
r2_values = np.linspace(1, 50, 50)  # Firing rates for neuron 2 (Hz)
num_samples = 1000  # Number of samples to simulate


# Function to simulate spike counts
def simulate_spike_counts(rate, T, num_samples):
    lambda_param = rate * T
    return np.random.poisson(lambda_param, num_samples)


# Initialize accuracy matrix
accuracy_matrix = np.zeros((len(r1_values), len(r2_values)))

# Iterate over firing rates
for i, r1 in enumerate(r1_values):
    for j, r2 in enumerate(r2_values):
        # Simulate spike counts for both neurons
        spikes_neuron1 = simulate_spike_counts(r1, T, num_samples)
        spikes_neuron2 = simulate_spike_counts(r2, T, num_samples)

        # Combine data and labels
        data = np.concatenate((spikes_neuron1, spikes_neuron2))
        labels = np.concatenate((np.zeros(num_samples), np.ones(num_samples)))

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, data)
        roc_auc = auc(fpr, tpr)

        # Find the optimal threshold where slope is closest to 1
        slopes = tpr / fpr  # Slope of the ROC curve
        optimal_idx = np.argmin(np.abs(slopes - 1))
        optimal_threshold = thresholds[optimal_idx]

        # Compute classification accuracy at the optimal threshold
        predictions = data >= optimal_threshold
        accuracy = np.mean(predictions == labels)
        accuracy_matrix[i, j] = accuracy

# Plotting the phase space
plt.figure(figsize=(10, 8))
plt.contourf(r1_values, r2_values, accuracy_matrix.T, levels=50, cmap="viridis")
plt.colorbar(label="Classification Accuracy")
plt.xlabel("Firing Rate of Neuron 1 (Hz)")
plt.ylabel("Firing Rate of Neuron 2 (Hz)")
plt.title("Classification Accuracy in Phase Space")

# Mark regions where accuracy surpasses 95%
contour = plt.contour(
    r1_values,
    r2_values,
    accuracy_matrix.T,
    levels=[0.95],
    colors="red",
    linestyles="--",
)
plt.clabel(contour, fmt="Accuracy = %.2f", colors="red")

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Function to generate a Poisson spike count over time T with bins of width dt
def generate_poisson_spike_counts(rate, T=1.0, dt=0.01, trials=1):
    """
    Generates spike counts for a Poisson neuron firing at a given rate.

    Args:
        rate (float): Firing rate (Hz)
        T (float): Total duration (seconds)
        dt (float): Bin width (seconds)
        trials (int): Number of independent trials

    Returns:
        np.array: Total spike count per trial
    """
    num_bins = int(T / dt)
    spike_counts = np.random.poisson(rate * dt, size=(trials, num_bins)).sum(axis=1)
    return spike_counts


# Function to compute classification accuracy based on thresholding
def compute_accuracy(r1, r2, T=1.0, dt=0.01, trials=1):
    """
    Computes classification accuracy for neurons firing at rates r1 and r2.

    Args:
        r1 (float): Firing rate of neuron 1 (Hz)
        r2 (float): Firing rate of neuron 2 (Hz)
        T (float): Duration of experiment (seconds)
        dt (float): Time bin size (seconds)
        trials (int): Number of samples per classification decision

    Returns:
        float: Classification accuracy
    """
    # Generate spike counts for both neurons
    spikes_1 = generate_poisson_spike_counts(r1, T, dt, trials)
    spikes_2 = generate_poisson_spike_counts(r2, T, dt, trials)

    # Ground truth labels: Neuron 1 → 0, Neuron 2 → 1
    labels = np.concatenate([np.zeros(trials), np.ones(trials)])

    # Predictions: Classify based on threshold (midpoint of r1 and r2)
    threshold = (r1 + r2) / 2
    predictions = np.concatenate([spikes_1 >= threshold, spikes_2 >= threshold])

    # Compute accuracy
    return accuracy_score(labels, predictions)


# Define the range of firing rates
r1_values = np.linspace(1, 50, 30)  # Firing rates for neuron 1 (1Hz to 50Hz)
r2_values = np.linspace(1, 50, 30)  # Firing rates for neuron 2

# Create grids to store accuracy values
accuracy_one_sample = np.zeros((len(r1_values), len(r2_values)))
accuracy_two_samples = np.zeros((len(r1_values), len(r2_values)))

# Compute accuracy for different rate pairs (r1, r2)
for i, r1 in enumerate(r1_values):
    for j, r2 in enumerate(r2_values):
        if r1 < r2:  # Ensure neuron 2 has a higher rate
            accuracy_one_sample[i, j] = compute_accuracy(r1, r2, trials=1)
            accuracy_two_samples[i, j] = compute_accuracy(r1, r2, trials=2)
        else:
            accuracy_one_sample[i, j] = np.nan  # Mark invalid cases
            accuracy_two_samples[i, j] = np.nan

# Plot classification accuracy for one-sample decision
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(
    accuracy_one_sample,
    extent=[1, 50, 1, 50],
    origin="lower",
    aspect="auto",
    cmap="hot",
    vmin=0.5,
    vmax=1,
)
plt.colorbar(label="Classification Accuracy")
plt.contour(
    r1_values,
    r2_values,
    accuracy_one_sample.T,
    levels=[0.95],
    colors="cyan",
    linewidths=2,
)
plt.xlabel("Neuron 1 Firing Rate (Hz)")
plt.ylabel("Neuron 2 Firing Rate (Hz)")
plt.title("Accuracy (1 Sample Decision)")

# Plot classification accuracy for two-sample decision
plt.subplot(1, 2, 2)
plt.imshow(
    accuracy_two_samples,
    extent=[1, 50, 1, 50],
    origin="lower",
    aspect="auto",
    cmap="hot",
    vmin=0.5,
    vmax=1,
)
plt.colorbar(label="Classification Accuracy")
plt.contour(
    r1_values,
    r2_values,
    accuracy_two_samples.T,
    levels=[0.95],
    colors="cyan",
    linewidths=2,
)
plt.xlabel("Neuron 1 Firing Rate (Hz)")
plt.ylabel("Neuron 2 Firing Rate (Hz)")
plt.title("Accuracy (2 Sample Decision)")

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Simulation Parameters
T = 1.0  # Time interval in seconds
r1_values = np.linspace(1, 50, 50)  # Firing rates for neuron 1 (Hz)
r2_values = np.linspace(1, 50, 50)  # Firing rates for neuron 2 (Hz)
num_samples = 1000  # Number of trials per neuron


# Function to generate spike counts for Poisson neurons
def simulate_spike_counts(rate, T, num_samples):
    return np.random.poisson(rate * T, num_samples)


# Function to compute accuracy given a decision threshold
def compute_accuracy(data, labels, threshold):
    predictions = data >= threshold
    return np.mean(predictions == labels)


# Initialize accuracy matrices for 1 sample and 2 sample cases
accuracy_matrix_1sample = np.zeros((len(r1_values), len(r2_values)))
accuracy_matrix_2samples = np.zeros((len(r1_values), len(r2_values)))

# Iterate over different firing rates
for i, r1 in enumerate(r1_values):
    for j, r2 in enumerate(r2_values):
        # Generate spike counts
        spikes_neuron1 = simulate_spike_counts(r1, T, num_samples)
        spikes_neuron2 = simulate_spike_counts(r2, T, num_samples)

        # Combine data and labels
        data = np.concatenate((spikes_neuron1, spikes_neuron2))
        labels = np.concatenate((np.zeros(num_samples), np.ones(num_samples)))

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, data)
        roc_auc = auc(fpr, tpr)

        # Find optimal threshold where slope of ROC curve is closest to 1
        slopes = np.divide(
            tpr, fpr, out=np.full_like(tpr, np.inf), where=fpr != 0
        )  # Avoid division by zero
        optimal_idx = np.argmin(np.abs(slopes - 1))
        optimal_threshold = thresholds[optimal_idx]

        # Compute classification accuracy
        accuracy_1sample = compute_accuracy(data, labels, optimal_threshold)
        accuracy_2samples = compute_accuracy(
            data[:num_samples] + data[num_samples:],
            labels[:num_samples],
            optimal_threshold * 2,
        )

        accuracy_matrix_1sample[i, j] = accuracy_1sample
        accuracy_matrix_2samples[i, j] = accuracy_2samples

# Plot classification accuracy phase space for 1 sample
plt.figure(figsize=(10, 8))
plt.contourf(r1_values, r2_values, accuracy_matrix_1sample.T, levels=50, cmap="viridis")
plt.colorbar(label="Classification Accuracy (1 sample)")
plt.xlabel("Firing Rate of Neuron 1 (Hz)")
plt.ylabel("Firing Rate of Neuron 2 (Hz)")
plt.title("Phase Space: Accuracy for 1 Sample")

# Mark regions where accuracy surpasses 95%
contour = plt.contour(
    r1_values,
    r2_values,
    accuracy_matrix_1sample.T,
    levels=[0.95],
    colors="red",
    linestyles="--",
)
plt.clabel(contour, fmt="Accuracy = %.2f", colors="red")

plt.show()

# Plot classification accuracy phase space for 2 samples
plt.figure(figsize=(10, 8))
plt.contourf(r1_values, r2_values, accuracy_matrix_2samples.T, levels=50, cmap="plasma")
plt.colorbar(label="Classification Accuracy (2 samples)")
plt.xlabel("Firing Rate of Neuron 1 (Hz)")
plt.ylabel("Firing Rate of Neuron 2 (Hz)")
plt.title("Phase Space: Accuracy for 2 Samples")

# Mark regions where accuracy surpasses 95%
contour = plt.contour(
    r1_values,
    r2_values,
    accuracy_matrix_2samples.T,
    levels=[0.95],
    colors="red",
    linestyles="--",
)
plt.clabel(contour, fmt="Accuracy = %.2f", colors="red")

plt.show()

# %%


def compute_roc_weighted(samples1, samples2, weights1, weights2):
    """
    Compute ROC curve for two weighted sample distributions.

    Parameters:
    - samples1 (np.array): Samples from first distribution (negative class, no spike)
    - samples2 (np.array): Samples from second distribution (positive class, spike)
    - weights1 (np.array): Frequency of each sample in samples1
    - weights2 (np.array): Frequency of each sample in samples2

    Returns:
    - fp (np.array): False positive rate
    - tp (np.array): True positive rate
    - thresholds (np.array): Threshold values used
    """
    # Combine all unique sample values for thresholding
    all_samples = np.concatenate((samples1, samples2))
    thresholds = np.sort(np.unique(all_samples))  # Unique threshold values

    # Ground truth labels (0 for samples1, 1 for samples2)
    y_true = np.concatenate((np.zeros(len(samples1)), np.ones(len(samples2))))
    sample_weights = np.concatenate((weights1, weights2))  # Merge weights

    # Initialize FP and TP arrays
    fp = np.zeros(len(thresholds))
    tp = np.zeros(len(thresholds))

    # Compute FP and TP for each threshold
    for i, z in enumerate(thresholds):
        y_pred = (all_samples >= z).astype(int)  # Predict spike if sample ≥ z

        # Compute weighted confusion matrix elements
        TP = np.sum(sample_weights[(y_pred == 1) & (y_true == 1)])  # True positives
        FP = np.sum(sample_weights[(y_pred == 1) & (y_true == 0)])  # False positives
        TN = np.sum(sample_weights[(y_pred == 0) & (y_true == 0)])  # True negatives
        FN = np.sum(sample_weights[(y_pred == 0) & (y_true == 1)])  # False negatives

        # Compute rates
        tp[i] = TP / (TP + FN) if (TP + FN) > 0 else 0  # True positive rate
        fp[i] = FP / (FP + TN) if (FP + TN) > 0 else 0  # False positive rate

    return fp, tp, thresholds


# Generate true tabulated Poisson distributions
np.random.seed(42)
T = 10000  # Large number for tabulated distribution
r1, r2 = 3, 7  # Firing rates for two neurons

samples1_tabulated, counts1 = np.unique(
    np.random.poisson(lam=r1, size=T), return_counts=True
)
samples2_tabulated, counts2 = np.unique(
    np.random.poisson(lam=r2, size=T), return_counts=True
)

# Normalize counts to be used as weights
weights1_tabulated = counts1 / np.sum(counts1)
weights2_tabulated = counts2 / np.sum(counts2)

# Compute ROC for tabulated distributions
fp_tabulated, tp_tabulated, _ = compute_roc_weighted(
    samples1_tabulated, samples2_tabulated, weights1_tabulated, weights2_tabulated
)

# Now sample smaller datasets
sample_sizes = [50, 200, 1000]
roc_results = []

for size in sample_sizes:
    samples1_sampled = np.random.poisson(lam=r1, size=size)
    samples2_sampled = np.random.poisson(lam=r2, size=size)

    # Compute ROC for sampled distributions (assuming uniform weight 1 per sample)
    weights1_sampled = np.ones_like(samples1_sampled)
    weights2_sampled = np.ones_like(samples2_sampled)

    fp_sampled, tp_sampled, _ = compute_roc_weighted(
        samples1_sampled, samples2_sampled, weights1_sampled, weights2_sampled
    )

    roc_results.append((fp_sampled, tp_sampled, f"Sampled (N={size})"))

# Plot comparison
plt.figure(figsize=(7, 7))
plt.plot(
    fp_tabulated,
    tp_tabulated,
    marker="o",
    linestyle="-",
    color="black",
    label="Tabulated (T=10000)",
)

for fp_sampled, tp_sampled, label in roc_results:
    plt.plot(fp_sampled, tp_sampled, linestyle="--", marker="o", label=label)

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier line
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve: Tabulated vs. Sampled Poisson Distributions")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# %%
