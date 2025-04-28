import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc

# %%
# Sample Input Data
input_array_1 = np.array([0, 1, 2, 3, 4, 5])
input_array_2 = np.array([10, 11, 12, 13, 14, 15])

# Prepare Data for Training
data = np.concatenate((input_array_1, input_array_2)).reshape(-1, 1)  # Reshape to 2D
labels = np.array([0] * len(input_array_1) + [1] * len(input_array_2))

# Split into Train and Test Sets
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.33, random_state=42)

# Define Threshold
threshold_z = 0.5  # determines how y_pred_probs is converted to y_pred_label (0 or 1)

# ---------------
# ROC Analysis Class
# ---------------
class ROCAnalysis:
    def __init__(self, data_train, labels_train, threshold_z):
        """
        Initialize the class with input arrays and threshold value.
        """
        self.data_train = np.array(data_train)
        self.labels_train = np.array(labels_train)
        self.threshold_z = threshold_z  # Decision threshold
        self.model = LogisticRegression()  # Define model but not trained yet

    def fitmodel(self):
        """
        Train the logistic regression model and predict probabilities.
        """
        self.model.fit(self.data_train, self.labels_train)  # Train model
        self.y_pred_probs = self.model.predict_proba(data_test)[:, 1]  # Probabilities for class 1
        return self.y_pred_probs

    def compute_confusion_matrix(self, y_pred_probs):
        """
        Compute the confusion matrix based on the selected threshold.
        """
        y_pred_label = (y_pred_probs >= self.threshold_z).astype(int)  # Apply threshold

        self.TP = np.sum((y_pred_label == 1) & (labels_test == 1))  # True Positives
        self.FP = np.sum((y_pred_label == 1) & (labels_test == 0))  # False Positives
        self.TN = np.sum((y_pred_label == 0) & (labels_test == 0))  # True Negatives
        self.FN = np.sum((y_pred_label == 0) & (labels_test == 1))  # False Negatives
        return y_pred_label, self.TP, self.FP, self.TN, self.FN 

    def calc_roc_auroc(self, y_pred_probs):
        """
        Compute the ROC curve and AUC score.
        """
        fpr, tpr, thresholds = roc_curve(labels_test, y_pred_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """
        Plot the ROC curve.
        """
        plt.figure(figsize=(6, 6))
        plt.plot(
            fpr,
            tpr,
            marker="o",
            linestyle="-",
            color="blue",
            label=f"ROC curve (AUC = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.3)  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

# ---------------
# Run the Analysis
# ---------------
roc = ROCAnalysis(data_train, labels_train, threshold_z)

# Train the model and get predicted probabilities
y_pred_probs = roc.fitmodel()

# Compute confusion matrix
y_pred_label, tp, fp, tn, fn = roc.compute_confusion_matrix(y_pred_probs)

# Compute ROC and AUC
fpr, tpr, thresholds, roc_auc = roc.calc_roc_auroc(y_pred_probs)

# Plot ROC Curve
roc.plot_roc_curve(fpr, tpr, roc_auc)

# %%
# Function to generate spikes
def generate_poisson_spikes(rate, T):
    inter_spike_intervals = np.random.exponential(1/rate, size=int(rate * T * 2))  # Generate more than needed
    spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times
    spike_times = spike_times[spike_times <= T]  # Keep only valid spike times
    return spike_times

# Parameters
r1 = 10  # Firing rate of neuron 1 (spikes per second)
r2 = 20  # Firing rate of neuron 2 (spikes per second)
T = 5    # Total time interval (seconds)

# Generate spikes for both neurons
spike_times_neuron1 = generate_poisson_spikes(r1, T)
spike_times_neuron2 = generate_poisson_spikes(r2, T)

# Create a dataset of features (spike counts)
num_samples = 1000
data = np.zeros((num_samples, 2))

for i in range(num_samples):
    # Generate a random time point
    time_point = np.random.uniform(0, T)
    # Count spikes in the time interval [0, time_point] for both neurons
    count1 = np.sum(spike_times_neuron1 <= time_point)
    count2 = np.sum(spike_times_neuron2 <= time_point)
    data[i] = [count1, count2]

# Create labels: label 1 for neuron 1 spikes and label 0 for neuron 2 spikes
labels = np.zeros(num_samples)
labels[data[:, 0] > data[:, 1]] = 1  # Label as 1 if neuron 1 has more spikes

# Initialize ROCAnalysis
threshold_z = 0.5  # Example threshold
roc_analysis = ROCAnalysis(data, labels, threshold_z)

# Fit the model and predict probabilities
y_pred_probs = roc_analysis.fitmodel(data)

# Compute confusion matrix
y_pred_label, TP, FP, TN, FN = roc_analysis.compute_confusion_matrix(y_pred_probs, labels)

# Calculate accuracy
accuracy = roc_analysis.compute_accuracy(y_pred_probs, labels)

# Calculate ROC and AUC
fpr, tpr, thresholds, roc_auc = roc_analysis.calc_roc_auroc(y_pred_probs, labels)

# Plot the ROC curve
roc_analysis.plot_roc_curve(fpr, tpr, roc_auc)

# Print results
print(f"Confusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC: {roc_auc:.2f}")

# %%

class ROCAnalysis:
    def __init__(self, data_train, labels_train, threshold_z):
        self.data_train = np.array(data_train)
        self.labels_train = np.array(labels_train)
        self.threshold_z = threshold_z  # Decision threshold
        self.model = LogisticRegression()  # Define model but not trained yet

    def fitmodel(self):
        """Train the logistic regression model and predict probabilities."""
        self.model.fit(self.data_train, self.labels_train)  # Train model
        self.y_pred_probs = self.model.predict_proba(self.data_train)[:, 1]  # Probabilities for class 1
        return self.y_pred_probs

    def compute_confusion_matrix(self, y_pred_probs, labels_test):
        """Compute the confusion matrix based on the selected threshold."""
        y_pred_label = (y_pred_probs >= self.threshold_z).astype(int)  # Apply threshold
        self.TP = np.sum((y_pred_label == 1) & (labels_test == 1))  # True Positives
        self.FP = np.sum((y_pred_label == 1) & (labels_test == 0))  # False Positives
        self.TN = np.sum((y_pred_label == 0) & (labels_test == 0))  # True Negatives
        self.FN = np.sum((y_pred_label == 0) & (labels_test == 1))  # False Negatives
        return y_pred_label, self.TP, self.FP, self.TN, self.FN 

    def compute_accuracy(self, y_pred_probs, labels_test):
        """Compute classification accuracy based on the decision threshold."""
        y_pred_label = (y_pred_probs >= self.threshold_z).astype(int)  # Apply threshold
        accuracy = np.mean(y_pred_label == labels_test)  # Calculate accuracy
        return accuracy

    def calc_roc_auroc(self, y_pred_probs, labels_test):
        """Compute the ROC curve and AUC score."""
        fpr, tpr, thresholds = roc_curve(labels_test, y_pred_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)  # Area under the ROC curve
        return fpr, tpr, thresholds, roc_auc

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot the ROC curve."""
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, marker="o", linestyle="-", color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.3)  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

# Function to generate spikes
def generate_poisson_spikes(rate, T):
    """Generate spike times for a Poisson neuron with a given firing rate."""
    inter_spike_intervals = np.random.poisson(1/rate, size=int(rate * T * 2))  # Generate more than needed
    spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times
    spike_times = spike_times[spike_times <= T]  # Keep only valid spike times
    return spike_times

# Parameters
r1 = 10  # Firing rate of neuron 1 (spikes per second)
r2 = 20  # Firing rate of neuron 2 (spikes per second)
T = 5    # Total time interval (seconds)

# Generate spikes for both neurons
spike_times_neuron1 = generate_poisson_spikes(r1, T)
spike_times_neuron2 = generate_poisson_spikes(r2, T)

# Create a dataset of features (spike counts)
num_samples = 1000
data = np.zeros((num_samples, 2))

for i in range(num_samples):
    # Generate a random time point
    time_point = np.random.uniform(0, T)
    # Count spikes in the time interval [0, time_point] for both neurons
    count1 = np.sum(spike_times_neuron1 <= time_point)
    count2 = np.sum(spike_times_neuron2 <= time_point)
    data[i] = [count1, count2]

# Create labels: label 1 for neuron 1 spikes and label 0 for neuron 2 spikes
labels = np.zeros(num_samples)
labels[data[:, 0] > data[:, 1]] = 1  # Label as 1 if neuron 1 has more spikes

# Initialize ROCAnalysis
threshold_z = 0.5  # Example threshold
roc_analysis = ROCAnalysis(data, labels, threshold_z)

# Fit the model and predict probabilities
y_pred_probs = roc_analysis.fitmodel()

# Compute confusion matrix
y_pred_label, TP, FP, TN, FN = roc_analysis.compute_confusion_matrix(y_pred_probs, labels)

# Calculate accuracy
accuracy = roc_analysis.compute_accuracy(y_pred_probs, labels)

# Calculate ROC and AUC
fpr, tpr, thresholds, roc_auc = roc_analysis.calc_roc_auroc(y_pred_probs, labels)

# Plot the ROC curve
roc_analysis.plot_roc_curve(fpr, tpr, roc_auc)

# Print results
print(f"Confusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC: {roc_auc:.2f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

class ROCAnalysis:
    def __init__(self, data_train, labels_train, threshold_z):
        self.data_train = np.array(data_train)
        self.labels_train = np.array(labels_train)
        self.threshold_z = threshold_z  # Decision threshold
        self.model = LogisticRegression()  # Define model but not trained yet

    def fitmodel(self):
        """Train the logistic regression model and predict probabilities."""
        self.model.fit(self.data_train, self.labels_train)  # Train model
        self.y_pred_probs = self.model.predict_proba(self.data_train)[:, 1]  # Probabilities for class 1
        return self.y_pred_probs

    def compute_confusion_matrix(self, y_pred_probs, labels_test):
        """Compute the confusion matrix based on the selected threshold."""
        y_pred_label = (y_pred_probs >= self.threshold_z).astype(int)  # Apply threshold
        self.TP = np.sum((y_pred_label == 1) & (labels_test == 1))  # True Positives
        self.FP = np.sum((y_pred_label == 1) & (labels_test == 0))  # False Positives
        self.TN = np.sum((y_pred_label == 0) & (labels_test == 0))  # True Negatives
        self.FN = np.sum((y_pred_label == 0) & (labels_test == 1))  # False Negatives
        return y_pred_label, self.TP, self.FP, self.TN, self.FN 

    def compute_accuracy(self, y_pred_probs, labels_test):
        """Compute classification accuracy based on the decision threshold."""
        y_pred_label = (y_pred_probs >= self.threshold_z).astype(int)  # Apply threshold
        accuracy = np.mean(y_pred_label == labels_test)  # Calculate accuracy
        return accuracy

    def calc_roc_auroc(self, y_pred_probs, labels_test):
        """Compute the ROC curve and AUC score."""
        fpr, tpr, thresholds = roc_curve(labels_test, y_pred_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)  # Area under the ROC curve
        return fpr, tpr, thresholds, roc_auc

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot the ROC curve."""
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, marker="o", linestyle="-", color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.3)  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

# Function to generate spikes
def generate_poisson_spikes(rate, T):
    """Generate spike times for a Poisson neuron with a given firing rate."""
    inter_spike_intervals = np.random.exponential(1/rate, size=int(rate * T * 2))  # Generate more than needed
    spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times
    spike_times = spike_times[spike_times <= T]  # Keep only valid spike times
    return spike_times

# Parameters
r1 = 10  # Firing rate of neuron 1 (spikes per second)
r2 = 20  # Firing rate of neuron 2 (spikes per second)
T = 5    # Total time interval (seconds)

# Generate spikes for both neurons
spike_times_neuron1 = generate_poisson_spikes(r1, T)
spike_times_neuron2 = generate_poisson_spikes(r2, T)

# Create a dataset of features (spike counts)
num_samples = 1000
data = np.zeros((num_samples, 2))

for i in range(num_samples):
    # Generate a random time point
    time_point = np.random.uniform(0, T)
    # Count spikes in the time interval [0, time_point] for both neurons
    count1 = np.sum(spike_times_neuron1 <= time_point)
    count2 = np.sum(spike_times_neuron2 <= time_point)
    data[i] = [count1, count2]

# Create labels based on the firing rates rather than simple counts
labels = np.random.choice([0, 1], size=num_samples, p=[r1/(r1 + r2), r2/(r1 + r2)])

# Initialize ROCAnalysis
threshold_z = 0.5  # Example threshold
roc_analysis = ROCAnalysis(data, labels, threshold_z)

# Fit the model and predict probabilities
y_pred_probs = roc_analysis.fitmodel()

# Compute confusion matrix
y_pred_label, TP, FP, TN, FN = roc_analysis.compute_confusion_matrix(y_pred_probs, labels)

# Calculate accuracy
accuracy = roc_analysis.compute_accuracy(y_pred_probs, labels)

# Calculate ROC and AUC
fpr, tpr, thresholds, roc_auc = roc_analysis.calc_roc_auroc(y_pred_probs, labels)

# Plot the ROC curve
roc_analysis.plot_roc_curve(fpr, tpr, roc_auc)

# Print results
print(f"Confusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC: {roc_auc:.2f}")

# %%

# Function to generate spikes for Poisson neurons
def generate_poisson_spikes(rate, T):
    """Generate spike times for a Poisson neuron with a given firing rate."""
    inter_spike_intervals = np.random.exponential(1/rate, size=int(rate * T * 2))  # Generate inter-spike intervals
    spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times
    spike_times = spike_times[spike_times <= T]  # Keep only valid spike times
    return spike_times

# Function to calculate accuracy and AUC
def calculate_accuracy_and_auc(r1, r2, num_samples=1000, T=5, threshold_z=0.5):
    # Generate spikes for both neurons
    spike_times_neuron1 = generate_poisson_spikes(r1, T)
    spike_times_neuron2 = generate_poisson_spikes(r2, T)

    # Create a dataset of features (spike counts)
    data = np.zeros((num_samples, 2))
    labels = np.random.choice([0, 1], size=num_samples, p=[r1/(r1 + r2), r2/(r1 + r2)])

    for i in range(num_samples):
        time_point = np.random.uniform(0, T)
        count1 = np.sum(spike_times_neuron1 <= time_point)
        count2 = np.sum(spike_times_neuron2 <= time_point)
        data[i] = [count1, count2]

    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(data, labels)
    y_pred_probs = model.predict_proba(data)[:, 1]  # Probabilities for class 1

    # Compute accuracy for one sample
    y_pred_labels = (y_pred_probs >= threshold_z).astype(int)
    accuracy = np.mean(y_pred_labels == labels)  # Classification accuracy

    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(labels, y_pred_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)  # Area under the ROC curve

    return accuracy, roc_auc

# Define parameters for firing rates
r1_values = np.linspace(1, 50, 100)  # Firing rates for neuron 1
r2_values = np.linspace(1, 50, 100)  # Firing rates for neuron 2

# Initialize accuracy and AUC arrays
accuracies = np.zeros((len(r1_values), len(r2_values)))
auroc_values = np.zeros((len(r1_values), len(r2_values)))

# Compute accuracy and AUC for each combination of firing rates
for i, r1 in enumerate(r1_values):
    for j, r2 in enumerate(r2_values):
        accuracy, roc_auc = calculate_accuracy_and_auc(r1, r2)
        accuracies[i, j] = accuracy
        auroc_values[i, j] = roc_auc

# Plot the results in phase space
plt.figure(figsize=(12, 6))

# Accuracy contour plot
plt.subplot(1, 2, 1)
plt.contourf(r1_values, r2_values, accuracies, levels=np.linspace(0, 1, 21), cmap='viridis', alpha=0.7)
plt.colorbar(label='Accuracy')
plt.contour(r1_values, r2_values, accuracies, levels=[0.95], colors='red', linewidths=2, linestyles='dashed')
plt.title('Classification Accuracy Phase Space')
plt.xlabel('Firing Rate of Neuron 1 (r1)')
plt.ylabel('Firing Rate of Neuron 2 (r2)')

# AUC contour plot
plt.subplot(1, 2, 2)
plt.contourf(r1_values, r2_values, auroc_values, levels=np.linspace(0, 1, 21), cmap='viridis', alpha=0.7)
plt.colorbar(label='AUC')
plt.title('AUC Phase Space')
plt.xlabel('Firing Rate of Neuron 1 (r1)')
plt.ylabel('Firing Rate of Neuron 2 (r2)')

plt.tight_layout()
plt.show()

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_curve, auc

# # Function to generate spikes for Poisson neurons
# def generate_poisson_spikes(rate, T):
#     """Generate spike times for a Poisson neuron with a given firing rate."""
#     inter_spike_intervals = np.random.exponential(1/rate, size=int(rate * T * 2))  # Generate inter-spike intervals
#     spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times
#     spike_times = spike_times[spike_times <= T]  # Keep only valid spike times
#     return spike_times

# # Function to calculate accuracy and AUC
# def calculate_accuracy_and_auc(r1, r2, num_samples=1000, T=5, threshold_z=0.5):
#     # Generate spikes for both neurons
#     spike_times_neuron1 = generate_poisson_spikes(r1, T)
#     spike_times_neuron2 = generate_poisson_spikes(r2, T)

#     # Create a dataset of features (spike counts)
#     data = np.zeros((num_samples, 2))
#     labels = np.zeros(num_samples)

#     for i in range(num_samples):
#         time_point = np.random.uniform(0, T)
#         count1 = np.sum(spike_times_neuron1 <= time_point)
#         count2 = np.sum(spike_times_neuron2 <= time_point)
#         data[i] = [count1, count2]
        
#         # Ensure labels reflect the neuron with a higher spike count
#         labels[i] = 1 if count2 > count1 else 0

#     # Ensure there are samples from both classes
#     if np.all(labels == 0) or np.all(labels == 1):
#         return 0.5, 0.5  # Return neutral values if no classes are present

#     # Fit logistic regression model
#     model = LogisticRegression()
#     model.fit(data, labels)
#     y_pred_probs = model.predict_proba(data)[:, 1]  # Probabilities for class 1

#     # Compute accuracy for one sample
#     y_pred_labels = (y_pred_probs >= threshold_z).astype(int)
#     accuracy = np.mean(y_pred_labels == labels)  # Classification accuracy

#     # Calculate ROC and AUC
#     fpr, tpr, _ = roc_curve(labels, y_pred_probs, pos_label=1)
#     roc_auc = auc(fpr, tpr)  # Area under the ROC curve

#     return accuracy, roc_auc

# # Example firing rates for better separation
# r1 = 5   # Firing rate of neuron 1 (spikes per second)
# r2 = 30  # Firing rate of neuron 2 (spikes per second)

# # Calculate accuracy and AUC
# accuracy, roc_auc = calculate_accuracy_and_auc(r1, r2)
# print(f"Accuracy: {accuracy:.2f}, AUC: {roc_auc:.2f}")

# # Optionally plot results in phase space as previously discussed.


# %%


# # Function to generate spikes for Poisson neurons
# def generate_poisson_spikes(rate, T):
#     """Generate spike times for a Poisson neuron with a given firing rate."""
#     inter_spike_intervals = np.random.exponential(1/rate, size=int(rate * T * 2))  # Generate inter-spike intervals
#     spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times
#     spike_times = spike_times[spike_times <= T]  # Keep only valid spike times
#     return spike_times

# # Function to calculate accuracy and AUC
# def calculate_accuracy_and_auc(r1, r2, num_samples=1000, T=5, threshold_z=0.5):
#     # Generate spikes for both neurons
#     spike_times_neuron1 = generate_poisson_spikes(r1, T)
#     spike_times_neuron2 = generate_poisson_spikes(r2, T)

#     # Create a dataset of features (spike counts)
#     data = np.zeros((num_samples, 2))
#     labels = np.zeros(num_samples)

#     for i in range(num_samples):
#         time_point = np.random.uniform(0, T)
#         count1 = np.sum(spike_times_neuron1 <= time_point) + np.random.normal(0, 1)  # Add noise
#         count2 = np.sum(spike_times_neuron2 <= time_point) + np.random.normal(0, 1)  # Add noise
#         data[i] = [count1, count2]

#         # Labeling with some overlap
#         if count2 > count1 + np.random.uniform(-2, 2):  # Allow some overlap
#             labels[i] = 1
#         else:
#             labels[i] = 0

#     # Ensure there are samples from both classes
#     if np.all(labels == 0) or np.all(labels == 1):
#         return 0.5, 0.5  # Return neutral values if no classes are present

#     # Fit logistic regression model
#     model = LogisticRegression()
#     model.fit(data, labels)
#     y_pred_probs = model.predict_proba(data)[:, 1]  # Probabilities for class 1

#     # Compute accuracy for one sample
#     y_pred_labels = (y_pred_probs >= threshold_z).astype(int)
#     accuracy = np.mean(y_pred_labels == labels)  # Classification accuracy

#     # Calculate ROC and AUC
#     fpr, tpr, _ = roc_curve(labels, y_pred_probs, pos_label=1)
#     roc_auc = auc(fpr, tpr)  # Area under the ROC curve

#     return accuracy, roc_auc

# Set firing rates closer together
r1 = 10  # Firing rate of neuron 1 (spikes per second)
r2 = 11  # Firing rate of neuron 2 (spikes per second)

# Calculate accuracy and AUC
accuracy, roc_auc = calculate_accuracy_and_auc(r1, r2)
print(f"Accuracy: {accuracy:.2f}, AUC: {roc_auc:.2f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Function to generate spikes for Poisson neurons
def generate_poisson_spikes(rate, T):
    """Generate spike times for a Poisson neuron with a given firing rate."""
    inter_spike_intervals = np.random.exponential(1/rate, size=int(rate * T * 2))  # Generate inter-spike intervals
    spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times
    spike_times = spike_times[spike_times <= T]  # Keep only valid spike times
    return spike_times

# Function to calculate accuracy and AUC
def calculate_accuracy_and_auc(r1, r2, num_samples=1000, T=5, threshold_z=0.5):
    # Generate spikes for both neurons
    spike_times_neuron1 = generate_poisson_spikes(r1, T)
    spike_times_neuron2 = generate_poisson_spikes(r2, T)

    # Create a dataset of features (spike counts)
    data = np.zeros((num_samples, 2))
    labels = np.zeros(num_samples)

    for i in range(num_samples):
        time_point = np.random.uniform(0, T)
        count1 = np.sum(spike_times_neuron1 <= time_point) + np.random.normal(0, 2)  # Increase noise
        count2 = np.sum(spike_times_neuron2 <= time_point) + np.random.normal(0, 2)  # Increase noise
        data[i] = [count1, count2]

        # Labeling with increased overlap
        if count2 > count1 + np.random.uniform(-10, 10):  # Increase overlap variability
            labels[i] = 1
        else:
            labels[i] = 0

    # Ensure there are samples from both classes
    if np.all(labels == 0) or np.all(labels == 1):
        return 0.5, 0.5  # Return neutral values if no classes are present

    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(data, labels)
    y_pred_probs = model.predict_proba(data)[:, 1]  # Probabilities for class 1

    # Compute accuracy for one sample
    y_pred_labels = (y_pred_probs >= threshold_z).astype(int)
    accuracy = np.mean(y_pred_labels == labels)  # Classification accuracy

    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(labels, y_pred_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)  # Area under the ROC curve

    return accuracy, roc_auc

# Set firing rates closer together
r1 = 10  # Firing rate of neuron 1 (spikes per second)
r2 = 10.5  # Firing rate of neuron 2 (spikes per second)

# Calculate accuracy and AUC
accuracy, roc_auc = calculate_accuracy_and_auc(r1, r2)
print(f"Accuracy: {accuracy:.2f}, AUC: {roc_auc:.2f}")


# Define parameters for firing rates
r1_values = np.linspace(1, 50, 100)  # Firing rates for neuron 1
r2_values = np.linspace(1, 50, 100)  # Firing rates for neuron 2

# Initialize accuracy and AUC arrays
accuracies = np.zeros((len(r1_values), len(r2_values)))
auroc_values = np.zeros((len(r1_values), len(r2_values)))

# Compute accuracy and AUC for each combination of firing rates
for i, r1 in enumerate(r1_values):
    for j, r2 in enumerate(r2_values):
        accuracy, roc_auc = calculate_accuracy_and_auc(r1, r2)
        accuracies[i, j] = accuracy
        auroc_values[i, j] = roc_auc

# Plot the results in phase space
plt.figure(figsize=(12, 6))

# Accuracy contour plot
plt.subplot(1, 2, 1)
plt.contourf(r1_values, r2_values, accuracies, levels=np.linspace(0, 1, 21), cmap='viridis', alpha=0.7)
plt.colorbar(label='Accuracy')
plt.contour(r1_values, r2_values, accuracies, levels=[0.95], colors='red', linewidths=2, linestyles='dashed')
plt.title('Classification Accuracy Phase Space')
plt.xlabel('Firing Rate of Neuron 1 (r1)')
plt.ylabel('Firing Rate of Neuron 2 (r2)')

# AUC contour plot
plt.subplot(1, 2, 2)
plt.contourf(r1_values, r2_values, auroc_values, levels=np.linspace(0, 1, 21), cmap='viridis', alpha=0.7)
plt.colorbar(label='AUC')
plt.title('AUC Phase Space')
plt.xlabel('Firing Rate of Neuron 1 (r1)')
plt.ylabel('Firing Rate of Neuron 2 (r2)')

plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Data Validator Class
class DataValidator:
    def __init__(self, input_arr_1, input_arr_2):
        self.input_arr_1 = input_arr_1
        self.input_arr_2 = input_arr_2

    def check_input_format(self):
        for idx, inp in enumerate((self.input_arr_1, self.input_arr_2)):
            if not isinstance(inp, (list, tuple, np.ndarray)):
                raise TypeError(f"Error: input_arr_{idx+1} must be a list, tuple, or numpy array, but got {type(inp)}")
        print("Valid: Both inputs are lists, tuples, or arrays.")

    def check_empty(self):
        for idx, inp in enumerate((self.input_arr_1, self.input_arr_2)):
            if len(inp) == 0:
                raise ValueError(f"input_arr_{idx+1} cannot be empty.")
        print("Valid: Both inputs are not empty.")

    def check_type(self):
        for idx, inp in enumerate((self.input_arr_1, self.input_arr_2)):
            if isinstance(inp, np.ndarray):
                if not np.issubdtype(inp.dtype, np.number):
                    raise TypeError(f"All elements in input_arr_{idx+1} must be numeric (int or float).")
            else:
                if not all(isinstance(item, (int, float, np.integer, np.floating)) for item in inp):
                    raise TypeError(f"All elements in input_arr_{idx+1} must be numeric (int or float).")
        print("Valid: All inputs contain good numbers.")

    def validate(self):
        self.check_input_format()
        self.check_empty()
        self.check_type()
        print("Input check complete: Computer says yes")
####
# ROC Analysis Class
class ROCAnalysis:
    def __init__(self, data_train, labels_train, threshold_z):
        self.data_train = np.array(data_train)
        self.labels_train = np.array(labels_train)
        self.threshold_z = threshold_z
        self.model = RandomForestClassifier()

    def fitmodel(self, data_test):
        """Train the logistic regression model and predict probabilities."""
        self.model.fit(self.data_train, self.labels_train)
        self.y_pred_probs = self.model.predict_proba(data_test)[:, 1]
        return self.y_pred_probs

    def comp_matrix(self, y_pred_probs, labels_test):
        y_pred_label = (y_pred_probs >= self.threshold_z).astype(int)
        self.TP = np.sum((y_pred_label == 1) & (labels_test == 1))
        self.FP = np.sum((y_pred_label == 1) & (labels_test == 0))
        self.TN = np.sum((y_pred_label == 0) & (labels_test == 0))
        self.FN = np.sum((y_pred_label == 0) & (labels_test == 1))
        return y_pred_label, self.TP, self.FP, self.TN, self.FN 

    def comp_accuracy(self, y_pred_probs, labels_test):
        y_pred_label = (y_pred_probs >= self.threshold_z).astype(int)
        accuracy = np.mean(y_pred_label == labels_test)
        return accuracy

    def calc_roc_auroc(self, y_pred_probs, labels_test):
        """Compute ROC curve and AUROC score."""
        fpr, tpr, thresholds = roc_curve(labels_test, y_pred_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot the ROC curve."""
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, marker="o", linestyle="-", color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.3)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

# Function to prepare data
def prepare_data(input_arr_1, input_arr_2):
    # Validate inputs
    validator = DataValidator(input_arr_1, input_arr_2)
    validator.validate()

    # Combine data and assign labels
    data_train = np.concatenate((input_arr_1, input_arr_2)).reshape(-1, 1)
    labels = np.array([0] * len(input_arr_1) + [1] * len(input_arr_2))

    # Split into Train and Test Sets
    training_data, test_data, training_labels, test_labels = train_test_split(
        data_train, labels, test_size=0.33, random_state=42
    )

    return training_data, test_data, training_labels, test_labels

# Function to generate spikes for Poisson neurons
def generate_poisson_spikes(rate, T):
    """
    Generate spike times for a Poisson neuron with a given firing rate.
    
    Parameters:
    - rate: Firing rate (spikes per second)
    - T: Total time interval (in seconds)
    
    Returns:
    - spike_times: Array of spike times
    """
    inter_spike_intervals = np.random.exponential(1/rate, size=int(rate * T * 2))  # Generate more than needed
    spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times
    spike_times = spike_times[spike_times <= T]  # Keep only valid spike times
    return spike_times

# Convert spike times to binary format
def spikes_to_binary(spike_times, T):
    """Convert spike times to a binary representation."""
    time_bins = np.linspace(0, T, num=int(T * 100))  # 100 bins per second
    binary_spikes = np.zeros(len(time_bins) - 1)
    for spike in spike_times:
        bin_index = np.digitize(spike, time_bins) - 1
        if bin_index < len(binary_spikes):
            binary_spikes[bin_index] = 1
    return binary_spikes

# Function to calculate accuracy and AUC
def calculate_accuracy_and_auc(r1, r2, num_samples=1000, T=5, threshold_z=0.5):
    # Generate spikes for both neurons
    spike_times_neuron1 = generate_poisson_spikes(r1, T)
    spike_times_neuron2 = generate_poisson_spikes(r2, T)

    # Create a dataset of features (spike counts)
    data = np.zeros((num_samples, 2))
    labels = np.zeros(num_samples)

    for i in range(num_samples):
        time_point = np.random.uniform(0, T)
        count1 = np.sum(spike_times_neuron1 <= time_point) + np.random.normal(0, 2)
        count2 = np.sum(spike_times_neuron2 <= time_point) + np.random.normal(0, 2)
        data[i] = [count1, count2]

        # Labeling with increased overlap
        if count2 > count1 + np.random.uniform(-10, 10):
            labels[i] = 1
        else:
            labels[i] = 0

    # Ensure there are samples from both classes
    if np.all(labels == 0) or np.all(labels == 1):
        return 0.5, 0.5  # Return neutral values if no classes are present

    # Fit logistic regression model
    model = R()
    model.fit(data, labels)
    y_pred_probs = model.predict_proba(data)[:, 1]  # Probabilities for class 1

    # Compute accuracy for one sample
    y_pred_labels = (y_pred_probs >= threshold_z).astype(int)
    accuracy = np.mean(y_pred_labels == labels)  # Classification accuracy

    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(labels, y_pred_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)  # Area under the ROC curve

    return accuracy, roc_auc

# Parameters for firing rates
r1 = 30  # Firing rate of neuron 1 (spikes per second)
r2 = 50  # Firing rate of neuron 2 (spikes per second)
T = 5    # Total time interval (seconds)

# Generate spikes for both neurons
spike_times_neuron1 = generate_poisson_spikes(r1, T)
spike_times_neuron2 = generate_poisson_spikes(r2, T)

# Prepare binary spike data for ROC analysis
binary_spike_neuron1 = spikes_to_binary(spike_times_neuron1, T)
binary_spike_neuron2 = spikes_to_binary(spike_times_neuron2, T)

# Prepare data for analysis
data_train, data_test, train_labels, test_labels = prepare_data(binary_spike_neuron1, binary_spike_neuron2)

# Run ROC Analysis
thresh_z = 0.5  # Decision threshold
roc = ROCAnalysis(data_train, train_labels, thresh_z)

# Train the model and get predicted probabilities
y_pred_probs = roc.fitmodel(data_test)

# Compute confusion matrix
y_pred_label, tp, fp, tn, fn = roc.comp_matrix(y_pred_probs, test_labels)

# Calculate accuracy
accuracy = roc.comp_accuracy(y_pred_probs, test_labels)
print(f'Accuracy: {accuracy:.2f}')

# Compute ROC and AUC
fpr, tpr, thresholds, roc_auc = roc.calc_roc_auroc(y_pred_probs, test_labels)

# Plot ROC Curve
roc.plot_roc_curve(fpr, tpr, roc_auc)

# Plot the spike times
plt.figure(figsize=(10, 5))
plt.eventplot([spike_times_neuron1, spike_times_neuron2], colors=['blue', 'red'], linelengths=0.8)
plt.title('Spike Times of Two Poisson Neurons')
plt.xlabel('Time (seconds)')
plt.ylabel('Neurons')
plt.yticks([0, 1], ['Neuron 1 (r1)', 'Neuron 2 (r2)'])
plt.xlim(0, T)
plt.grid(True)
plt.show()

# Set firing rates for phase space analysis
r1_values = np.linspace(1, 50, 100)  # Firing rates for neuron 1
r2_values = np.linspace(1, 50, 100)  # Firing rates for neuron 2

# Initialize accuracy and AUC arrays
accuracies = np.zeros((len(r1_values), len(r2_values)))
auroc_values = np.zeros((len(r1_values), len(r2_values)))

# Compute accuracy and AUC for each combination of firing rates
for i, r1 in enumerate(r1_values):
    for j, r2 in enumerate(r2_values):
        accuracy, roc_auc = calculate_accuracy_and_auc(r1, r2)
        accuracies[i, j] = accuracy
        auroc_values[i, j] = roc_auc

# Plot the results in phase space
plt.figure(figsize=(12, 6))

# Accuracy contour plot
plt.subplot(1, 2, 1)
plt.contourf(r1_values, r2_values, accuracies, levels=np.linspace(0, 1, 21), cmap='viridis', alpha=0.7)
plt.colorbar(label='Accuracy')
plt.contour(r1_values, r2_values, accuracies, levels=[0.95], colors='red', linewidths=2, linestyles='dashed')
plt.title('Classification Accuracy Phase Space')
plt.xlabel('Firing Rate of Neuron 1 (r1)')
plt.ylabel('Firing Rate of Neuron 2 (r2)')

# AUC contour plot
plt.subplot(1, 2, 2)
plt.contourf(r1_values, r2_values, auroc_values, levels=np.linspace(0, 1, 21), cmap='viridis', alpha=0.7)
plt.colorbar(label='AUC')
plt.title('AUC Phase Space')
plt.xlabel('Firing Rate of Neuron 1 (r1)')
plt.ylabel('Firing Rate of Neuron 2 (r2)')

plt.tight_layout()
plt.show()

# %%
np.random.Generator.exponential(self, )