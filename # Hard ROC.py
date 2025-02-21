# Hard ROC


# The receiver-operator characteristics can be used to quantify how well you
#  can distinguish samples drawn from two different probability distributions.
# Your task is to implement the ROC analysis with your own Python code,
# and to quantify how well you can discriminate between the activity from
# two neurons firing according to a Poisson statistics

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %%
# data
# example distributions, check for errors (empty array, array input not string, list same length?, arrays the same?)
samples_t1 = "Schnubbel"  # not strings
samples_t2 = "Dubbel"
samples_a = np.array([])  # not empty
samples_b = np.array([0,])
samples_c = np.array([1,])
samples_d = np.array([0, 1, 1, 1, 1, 1])
samples_e = np.array([1, 1, 1, 1, 1, 2])
samples_f = np.array([1.001, 1.002, 1.003, 42000])
samples_g = np.array([1.0021, 1.0028, 1.0029, 1.0027])
samples_h = np.array([1, 1, 1, 1, 1, 1, 1, 1])
samples_i = np.array([1, 1])

# %%
# a) Write a function which takes two Numpy vectors samples1 and samples2 and
# computes the ROC curve, i.e. how false positives fp(z)
# and true positives tp(z) increase with decreasing decision threshold z
# (for background info see Decoding chapter in Computational Neurosciences script).
# Your function shall return two vectors fp and tp.
# For avoiding to lose information by having to choose an arbitrary bin width,
# do not compute histograms from the samples. The challenge for you is to
# rather compute the ROC curve directly from the sample vectors.
# Remember: Get a cafe if computing ROC is too hard!

# b vs c
# d vs e
# f vs g
# h vs 1
input = np.array([
    (samples_b, samples_c),
    (samples_d, samples_e),
    (samples_f, samples_g),
    (samples_h, samples_i)
], dtype=object)


def compute_roc_curve(samples1, samples2):

    # Combine samples and create corresponding labels
    scores = np.concatenate([samples1, samples2])
    labels = np.concatenate([np.zeros(len(samples1)), np.ones(len(samples2))])  # 0 = negatives, 1 = positives

    # Sort scores and labels in descending order
    sorted_indices = np.argsort(-scores) 
    sorted_labels = labels[sorted_indices]  

    
    tpr = np.cumsum(sorted_labels) / len(samples2)  
    fpr = np.cumsum(1 - sorted_labels) / len(samples1) 

    return fpr, tpr



def run_roc_analysis(input_array):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  
    axes = axes.ravel()  # Flatten to a 1D array for easy indexing
    
    for idx, (sample1, sample2) in enumerate(input_array):
            fpr, tpr = compute_roc_curve(sample1, sample2)  
            
            ax = axes[idx] 
            ax.plot(fpr, tpr, marker='o', label="ROC Curve")
            ax.plot([0, 1], [0, 1], 'k--', label="Random Guess") 
            ax.set_title(f"ROC Curve - Pair {idx+1}")
            ax.legend()

    plt.tight_layout()  
    plt.show()  


run_roc_analysis(input)

# %%
# error and exception handling

class DataValidator:
    def __init__(self, input_arr_1, input_arr_2):
        self.input_arr_1 = input_arr_1
        self.input_arr_2 = input_arr_2

    def check_input_format(self):
        for idx, inp in enumerate((self.input_arr_1, self.input_arr_2)):
            if not isinstance(inp, (list, tuple, np.ndarray)):
                raise TypeError(
                    f"Error: input_arr_{idx+1} must be a list, tuple, or numpy array, but got {type(inp)}"
                )
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
                    raise TypeError(
                        f"All elements in input_arr_{idx+1} must be numeric (int or float)."
                    )
            else:
                if not all(
                    isinstance(item, (int, float, np.integer, np.floating))
                    for item in inp
                ):
                    raise TypeError(
                        f"All elements in input_arr_{idx+1} must be numeric (int or float)."
                    )
        print("Valid: Good types only.")

    def validate(self):
        self.check_input_format()
        self.check_empty()
        self.check_type()
        print("Input check complete: Computer says yes")

# %%
# validate, split into training and test data, add labels 

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

# %%
# ROC Analysis Class
# train model, return probabilities 
# define what is true positive, true negative, false negative
# compute frp, trp 
# calc and plot roc, accuracy and auroc 

class ROCAnalysis:
    def __init__(self, data_train, labels_train, threshold_z):
        self.data_train = np.array(data_train)
        self.labels_train = np.array(labels_train)
        self.threshold_z = threshold_z
        self.model = LogisticRegression()
    
    def fitmodel(self, data_test):
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

# d) Extend your function by also computing and returning acc(z)
    def comp_accuracy(self, y_pred_label, labels_test):
        accuracy = np.mean(y_pred_label == labels_test)
        return accuracy
    
# c) Extend your function by also computing and returning the auroc
    def calc_roc_auroc(self, y_pred_probs, labels_test):
        fpr, tpr, thresholds = roc_curve(labels_test, y_pred_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot the ROC curve."""
        plt.figure(figsize=(6, 6))
        plt.plot(
            fpr,
            tpr,
            marker="o",
            linestyle="-",
            color="blue",
            label=f"ROC curve (AUC = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.3)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

# %% 
# # Function to generate spikes for Poisson neurons
def generate_poisson_spikes(rate, T):
    """
    generate spike times
    """
    inter_spike_intervals = np.random.exponential(1/rate, size=int(rate * T * 2))  # Generate more than needed
    spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times
    spike_times = spike_times[spike_times <= T]  # Keep only valid spike times
    return spike_times

# Convert spike times to binary format
def spikes_to_binary(spike_times, T):
    """Convert spike times to a binary representation."""
    time_bins = np.linspace(0, T, num=int(T * 1000))  # 1000 bins per second
    binary_spikes = np.zeros(len(time_bins) - 1)
    for spike in spike_times:
        bin_index = np.digitize(spike, time_bins) - 1
        if bin_index < len(binary_spikes):
            binary_spikes[bin_index] = 1
    return binary_spikes



# %% 

# Generate 2 arrays filled with spike trains

neuron1_data = []
neuron2_data = []
for i in range(500):
    spike_times_neuron1 = generate_poisson_spikes(r1, T)
    spike_times_neuron2 = generate_poisson_spikes(r2, T)
    binary_spike_neuron1 = spikes_to_binary(spike_times_neuron1, T)
    binary_spike_neuron2 = spikes_to_binary(spike_times_neuron2, T)
    neuron1_data.append(sum(binary_spike_neuron1))
    neuron2_data.append(sum(binary_spike_neuron2))

# apply data
data_train, data_test, train_labels, test_labels = prepare_data(
    neuron1_data, neuron2_data
)

# run ROC Analysis
thresh_z = 0.5  # Decision threshold
roc = ROCAnalysis(data_train, train_labels, thresh_z)

# train the model, get predictionss
y_pred_probs = roc.fitmodel(data_test)
print(y_pred_probs)
# Compute matrix
y_pred_label, tp, fp, tn, fn = roc.comp_matrix(y_pred_probs, test_labels)

# Calculate accuracy
accuracy = roc.comp_accuracy(y_pred_label, test_labels)
print(f"Accuracy: {accuracy:.2f}")

# Compute ROC and AUC
fpr, tpr, thresholds, roc_auc = roc.calc_roc_auroc(y_pred_label, test_labels)

# Plot ROC Curve
roc.plot_roc_curve(fpr, tpr, roc_auc)

plt.hist(y_pred_probs, bins=50)


# %%
# e) apply that to calc acc and auroc for different tresholds and plot 

def calculate_model(r1, r2, T=5, thresh_z=0.5):
    neuron1_data = []
    neuron2_data = []
    for i in range(100):
        spike_times_neuron1 = generate_poisson_spikes(r1, T)
        spike_times_neuron2 = generate_poisson_spikes(r2, T)
        binary_spike_neuron1 = spikes_to_binary(spike_times_neuron1, T)
        binary_spike_neuron2 = spikes_to_binary(spike_times_neuron2, T)
        neuron1_data.append(sum(binary_spike_neuron1))
        neuron2_data.append(sum(binary_spike_neuron2))

    # Prepare data for analysis
    data_train, data_test, train_labels, test_labels = prepare_data(neuron1_data, neuron2_data)

    # Run ROC Analysis
    roc = ROCAnalysis(data_train, train_labels, thresh_z)

    # Train the model and get predicted probabilities
    y_pred_probs = roc.fitmodel(data_test)

    # Compute matrix
    y_pred_label, tp, fp, tn, fn = roc.comp_matrix(y_pred_probs, test_labels)

    # Calculate accuracy
    accuracy = roc.comp_accuracy(y_pred_label, test_labels)
    print(f'Accuracy: {accuracy:.2f}')

    # Compute ROC and AUC
    fpr, tpr, thresholds, roc_auc = roc.calc_roc_auroc(y_pred_probs, test_labels)
    return accuracy, roc_auc


r1_values = range(30, 60)  # Firing rates for neuron 1
r2_values = range(30, 60) #np.linspace(30, 60)  # Firing rates for neuron 2

# Initialize accuracy and AUC arrays
accuracies = np.zeros((len(r1_values), len(r2_values)))
auroc_values = np.zeros((len(r1_values), len(r2_values)))

# Compute accuracy and AUC for each combination of firing rates
for i, r1 in enumerate(r1_values):
    for j, r2 in enumerate(r2_values):
        accuracy, roc_auc = calculate_model(r1, r2)
        accuracies[i, j] = accuracy
        auroc_values[i, j] = roc_auc


# Plot the results in phase space
plt.figure(figsize=(12, 6))

# Accuracy contour plot
plt.subplot(1, 2, 1)
plt.contourf(    r1_values,
    r2_values, 
    accuracies,
    levels=np.linspace(0, 1, 21),
    cmap="viridis",
    alpha=0.7,
)
plt.colorbar(label="Accuracy")
plt.contour(
    r1_values, 
    r2_values,
    accuracies,
    levels=[0.95],
    colors="red",
    linewidths=2,
    linestyles="dashed",
)
plt.title("Classification Accuracy Phase Space")
plt.xlabel("Firing Rate of Neuron 1 (r1)")
plt.ylabel("Firing Rate of Neuron 2 (r2)")

# AUC contour plot
plt.subplot(1, 2, 2)
plt.contourf(
    r1_values, 
    r2_values, 
    auroc_values,
    levels=np.linspace(0, 1, 21),
    cmap="viridis",
    alpha=0.7,
)
plt.colorbar(label="AUC")
plt.title("AUC Phase Space")
plt.xlabel("Firing Rate of Neuron 1 (r1)")
plt.ylabel("Firing Rate of Neuron 2 (r2)")

plt.tight_layout()
plt.show()
