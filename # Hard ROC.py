# Hard ROC
# A receiver operating characteristic curve, or ROC curve, 
# is a graphical plot that illustrates the performance of a 
# binary classifier model (can be used for multi class classification as well) 
# at varying threshold values.

# The ROC curve is the plot of the true positive rate (TPR) against
#  the false positive rate (FPR) at each threshold setting. 


# The receiver-operator characteristics can be used to quantify how well you
#  can distinguish samples drawn from two different probability distributions.
# Your task is to implement the ROC analysis with your own Python code,
# and to quantify how well you can discriminate between the activity from 
# two neurons firing according to a Poisson statistics

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %% 
# data
# example distributions, check for errors (empty array, array input not string, list same length?, arrays the same?)
samples_t1 = "Schnubbel"        # not strings
samples_t2 = "Dubbel"

samples_a = np.array([])                # not empty
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
# preparation: (label if not yet done), split data into training and test data

# %% 
# b vs c
# d vs e 
# f vs g 
# h vs 1

input_arr_1 =  samples_d    
input_arr_2 = samples_e
thresh_z = 4

# %% 
# Error and excption handling - validate input arrays

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
        print("Valid: Good types only.")

    def validate(self):
        self.check_input_format()
        self.check_empty()
        self.check_type()
        print("Input check complete: Computer says yes")


# def check_len(inp1, inp2):
#     if len(inp1) != len(inp2):  
#         raise ValueError("Input arrays need to be the same length")
#     else:
#         print("Valid length") 



# def check_pure_type(inp1, inp2):
#     """Ensure both inputs contain only integers or only floats.
#     If either contains floats or mixed types, ask the user whether to convert all to floats.
#     If the user refuses, raise a TypeError.
#     """

#     def get_type(lst):
#         """Returns 'int' if all elements are integers, 'float' if all are floats, and 'mixed' if both exist."""
#         has_ints = any(isinstance(item, (int, np.integer)) for item in lst)
#         has_floats = any(isinstance(item, (float, np.floating)) for item in lst)

#         if has_ints and has_floats:
#             return "mixed"
#         elif has_floats:
#             return "float"
#         elif has_ints:
#             return "int"
#         return "unknown"  # If empty or non-numeric values

#     def convert_to_floats(data):
#         """Convert a list/tuple/array to all floats while preserving the original format."""
#         converted = [float(item) for item in data]
#         if isinstance(data, tuple):
#             return tuple(converted)  # Preserve tuple format
#         elif isinstance(data, np.ndarray):
#             return np.array(converted, dtype=np.float64)  # Preserve NumPy array format
#         return converted  # Default to list

    # # Identify types
    # type1 = get_type(inp1)
    # type2 = get_type(inp2)

    # # Check if conversion is needed
    # needs_conversion = "float" in (type1, type2) or "mixed" in (type1, type2) or (type1 != type2)

    # if needs_conversion:
    #     response = input("One or both inputs contain floats or mixed types. Please check your input. Do you really want to convert all to floats? (y/n): ").strip().lower()

    #     if response == "yes":
    #         inp1, inp2 = convert_to_floats(inp1), convert_to_floats(inp2)
    #         print(f" Converted both inputs to floats:\n🔹 inp1: {inp1}\n🔹 inp2: {inp2}")
    #     else:
    #         raise TypeError(f" Type mismatch: inp1 contains '{type1}', inp2 contains '{type2}'. Cannot proceed without conversion.")

    # else:
    #     print("Valid input. No conversion needed.")

    # return inp1, inp2  # Return modified values


# %%
# ROC Analysis Class
class ROCAnalysis:
    def __init__(self, data_train, labels_train, threshold_z):
        self.data_train = np.array(data_train)
        self.labels_train = np.array(labels_train)
        self.threshold_z = threshold_z
        self.model = LogisticRegression()

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

# %%
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

# %%
# Function to generate spikes for Poisson neurons
def generate_poisson_spikes(rate, T):
    """
    Generate spike times for a Poisson neuron.
    
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
    time_bins = np.linspace(0, T, num=int(T * 1000))  # 1000 bins per second
    binary_spikes = np.zeros(len(time_bins) - 1)
    for spike in spike_times:
        bin_index = np.digitize(spike, time_bins) - 1
        if bin_index < len(binary_spikes):
            binary_spikes[bin_index] = 1
    return binary_spikes


# %%

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
        count1 = np.sum(spike_times_neuron1 <= time_point)
        count2 = np.sum(spike_times_neuron2 <= time_point)
        data[i] = [count1, count2]

        # Labeling
        if count2 > count1: #+ np.random.uniform(-1, 1):
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

# %%

# Parameters for firing rates
r1 = 50  # Firing rate of neuron 1 (spikes per second)
r2 = 90  # Firing rate of neuron 2 (spikes per second)
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
thresh_z = 0.8  # Decision threshold
roc = ROCAnalysis(data_train, train_labels, thresh_z)

# Train the model and get predicted probabilities
y_pred_probs = roc.fitmodel(data_test)

# Compute matrix
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

# %%

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

# # %% markdown

# # ROC 
# # a) Write a function which takes two Numpy vectors samples1 and samples2 and 
# # computes the ROC curve, i.e. how false positives fp(z)
# # and true positives tp(z) increase with decreasing decision threshold z 

# # Your function shall return two vectors fp and tp.
# # For avoiding to lose information by having to choose an arbitrary bin width,
# # do not compute histograms from the samples. The challenge for you is to 
# # rather compute the ROC curve directly from the sample vectors. 

# # input arrays

# # combine, add labels, split into train and test data
# # initialize prediction model

# # calculate TP, TN, FP, FN
# # calculate & return trp, fpr

# # calculate roc
# # calculate auroc
# # plot roc, show auroc


# # %% 
# #b) Test your functions on the examples provided in the code snippet roc samples.py. They should result
# # in the plots shown on the next page. Make sure that your code detects putative inconsistencies in the
# # input, e.g. wrong data types or empty sample vectors. Do some decent error catching if your assertions
# # fail!
# # compare the samples t1 versus t2, a versus b, b versus c, d versus e, f versus g,
# # h versus i. The first two comparisons should results in nicely caught errors, and then you would get the
# # following results for the remaining comparisons:
# # b vs c
# # d vs e 
# # f vs g 
# # h vs 1

# # %% 
# # d) Extend your function by also computing and returning acc(z). acc(z) should be the classification
# #accuracy acc given one sample, drawn with equal likelihood from one of the two distributions. This
# #measure depends on the chosen decision threshold z.

# # classification accuracy (depending on z, test for different z)

# # %%
# # e) Apply your function to spike counts generated by two ’Poisson’ neurons firing with constant rates r1
# #and r2, respectively, over a time interval T. Show in phase space how classification accuracy with one
# #sample (acc) or two samples (auroc) depend on the firing rates. Mark the boundary where accuracy
# #surpasses 95% correct.

# # generate spiking neurons based on poisson distr

# # classification accuracy one sample -95% correct
# #   with different firing rates

# # classification accuracy two samples - 95% correct
# # with different firing rates



# # %% 
# # f) JUST FOR FUN: Extend your ROC function to take two additional input vectors weights1 and
# # weights2 which specify how often a specific sample in samples1 and samples2 occur, respectively.
# # This gives you the opportunity to compute the ROC for a tabulated distribution function. Re-compute
# # the Poisson example for the tabulated distribution, and compare with the sampled distributions for
# # different number of samples drawn.