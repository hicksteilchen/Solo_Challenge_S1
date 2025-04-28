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
samples_t1 = "Schnubbel"  # not strings
samples_t2 = "Dubbel"

samples_a = np.array([])  # not empty
samples_b = np.array(
    [
        0,
    ]
)
samples_c = np.array(
    [
        1,
    ]
)

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

import numpy as np
from sklearn.model_selection import train_test_split


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
        print("Valid: All inputs contain good numbers.")

    def validate(self):
        self.check_input_format()
        self.check_empty()
        self.check_type()
        print("Input check complete: Computer says yes")


def prepare_data(input_arr_1, input_arr_2):
    # Validate inputs
    validator = DataValidator(input_arr_1, input_arr_2)
    validator.validate()

    # Combine data and assign labels
    data_train = np.concatenate((input_arr_1, input_arr_2)).reshape(-1, 1)
    labels = np.array(
        [0] * len(input_arr_1) + [1] * len(input_arr_2)
    )  # Assign labels: 0 for input_arr_1, 1 for input_arr_2

    # Split into Train and Test Sets
    training_data, test_data, training_labels, test_labels = train_test_split(
        data_train, labels, test_size=0.33, random_state=42
    )

    return training_data, test_data, training_labels, test_labels


# Example usage:
input_arr_1 = np.random.exponential(scale=1, size=100)  # Example input data for class 0
input_arr_2 = np.random.exponential(scale=1, size=100
)  # Example input data for class 1
thresh_z = 4  # Example threshold (not used in this part)

# Prepare the data
data_train, data_test, train_label, labels_test = prepare_data(input_arr_1, input_arr_2)

# Print results
print("Training data:\n", data_train)
print("Test data:\n", data_test)
print("Training labels:\n", train_label)
print("Test labels:\n", labels_test)


# %% markdown

# ROC
# a) Write a function which takes two Numpy vectors samples1 and samples2 and
# computes the ROC curve, i.e. how false positives fp(z)
# and true positives tp(z) increase with decreasing decision threshold z

# Your function shall return two vectors fp and tp.
# For avoiding to lose information by having to choose an arbitrary bin width,
# do not compute histograms from the samples. The challenge for you is to
# rather compute the ROC curve directly from the sample vectors.

# input arrays

# combine, add labels, split into train and test data
# initialize prediction model

# calculate TP, TN, FP, FN
# calculate & return trp, fpr

# calculate roc
# calculate auroc
# plot roc, show auroc

# data_train = np.concatenate((input_arr_1, input_arr_2))
# train_label = np.array([0] * len(input_arr_1) + [1] * len(input_arr_2))

# print(train_label)

# #validate_input(inp1, inp2)

# # %%
# # combine as data array
# # assign array_1 with label zero, array_2 with label 1 
# def split_data_labels(inp1, inp2):
#     data = np.concatenate((inp1, inp2)).reshape(-1, 1)
#     labels = np.array([0] * len(inp1) + [1] * len(inp2))  #inp1 gets label 0, inp2 gets label 1
# # Split into Train and Test Sets
#     trainingdata, testdata, traininglabel, testlabel = train_test_split(  # from sklearn.model_selection
#         data, labels, test_size=0.33, random_state=42
#         )
#     return trainingdata, testdata, traininglabel, testlabel

# data_train, data_test, train_label, labels_test = split_data_labels(input_arr_1, input_arr_2)
# print (data_train, data_test, train_label, labels_test)


# %%
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
        self.y_pred_probs = self.model.predict_proba(data_test)[
            :, 1
        ]  # Probabilities for class 1
        return self.y_pred_probs

    def comp_matrix(self, y_pred_probs):

        y_pred_label = (y_pred_probs >= self.threshold_z).astype(int)  # Apply threshold

        self.TP = np.sum((y_pred_label == 1) & (labels_test == 1))  # True Positives
        self.FP = np.sum((y_pred_label == 1) & (labels_test == 0))  # False Positives
        self.TN = np.sum((y_pred_label == 0) & (labels_test == 0))  # True Negatives
        self.FN = np.sum((y_pred_label == 0) & (labels_test == 1))  # False Negatives
        return y_pred_label, self.TP, self.FP, self.TN, self.FN

    # d) Extend your function by also computing and returning acc(z).
    def comp_accuracy(self, y_pred_probs, labels_test):

        y_pred_label = (y_pred_probs >= self.threshold_z).astype(int)  # Apply threshold
        accuracy = np.mean(y_pred_label == labels_test)  # Calculate accuracy
        return accuracy

    def calc_roc_auroc(self, y_pred_probs):
        """
        Compute ROC curve and auroc score.
        """
        fpr, tpr, thresholds = roc_curve(labels_test, y_pred_probs, pos_label=1)
        #        Extend your function by also computing and returning the auroc (i.e., area-under-the ROC curve). The
        # auroc provides the classification accuracy given two samples, with one taken from each distribution
        roc_auc = auc(fpr, tpr)  #
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
        plt.plot(
            [0, 1], [0, 1], color="gray", linestyle=":", alpha=0.3
        )  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()


# ---------------
# Run the Analysis
# ---------------
roc = ROCAnalysis(data_train, train_label, thresh_z)

# Train the model and get predicted probabilities
y_pred_probs = roc.fitmodel()

# Compute confusion matrix
y_pred_label, tp, fp, tn, fn = roc.comp_matrix(y_pred_probs)

accuracy = roc.comp_accuracy(y_pred_probs, labels_test)
print(accuracy)

# Compute ROC and AUC
fpr, tpr, thresholds, roc_auc = roc.calc_roc_auroc(y_pred_probs)


# Plot ROC Curve
roc.plot_roc_curve(fpr, tpr, roc_auc)


# %%
# b) Test your functions on the examples provided in the code snippet roc samples.py. They should result
# in the plots shown on the next page. Make sure that your code detects putative inconsistencies in the
# input, e.g. wrong data types or empty sample vectors. Do some decent error catching if your assertions
# fail!
# compare the samples t1 versus t2, a versus b, b versus c, d versus e, f versus g,
# h versus i. The first two comparisons should results in nicely caught errors, and then you would get the
# following results for the remaining comparisons:
# b vs c
# d vs e
# f vs g
# h vs 1

# %%
# d) Extend your function by also computing and returning acc(z). acc(z) should be the classification
# accuracy acc given one sample, drawn with equal likelihood from one of the two distributions. This
# measure depends on the chosen decision threshold z.

# classification accuracy (depending on z, test for different z)

# %%
# e) Apply your function to spike counts generated by two ’Poisson’ neurons firing with constant rates r1
# and r2, respectively, over a time interval T. Show in phase space how classification accuracy with one
# sample (acc) or two samples (auroc) depend on the firing rates. Mark the boundary where accuracy
# surpasses 95% correct.

# generate spiking neurons based on poisson distr

# classification accuracy one sample -95% correct
#   with different firing rates

# classification accuracy two samples - 95% correct
# with different firing rates


def generate_poisson_spikes(rate, T):
    """
    Generate spike times for a Poisson neuron with a given firing rate.

    Parameters:
    - rate: Firing rate (spikes per second)
    - T: Total time interval (in seconds)

    Returns:
    - spike_times: Array of spike times
    """
    # Generate inter-spike intervals
    inter_spike_intervals = np.random.exponential(
        1 / rate, size=int(rate * T * 2)
    )  # Generate more than needed
    # spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times

    # # Keep only the spike times that are within the time interval [0, T]
    # spike_times = spike_times[spike_times <= T]
    return inter_spike_intervals


# Parameters
r1 = 10  # Firing rate of neuron 1 (spikes per second)
r2 = 20  # Firing rate of neuron 2 (spikes per second)
T = 5  # Total time interval (seconds)

# Generate spikes for both neurons
interspike_times_neuron1 = generate_poisson_spikes(r1, T)
interspike_times_neuron2 = generate_poisson_spikes(r2, T)

plt.plot(interspike_times_neuron2)
plt.plot(interspike_times_neuron1)

# %% 
def generate_poisson_spikes(rate, T):
    # Generate inter-spike intervals
    inter_spike_intervals = np.random.exponential(
        1 / rate, size=int(rate * T * 2)
    )  # Generate more than needed
    spike_times = np.cumsum(inter_spike_intervals)  # Cumulative sum to get spike times

    # # Keep only the spike times that are within the time interval [0, T]
    spike_times = spike_times[spike_times <= T]
    return spike_times


# Parameters
r1 = 10  # Firing rate of neuron 1 (spikes per second)
r2 = 20  # Firing rate of neuron 2 (spikes per second)
T = 5  # Total time interval (seconds)

# Generate spikes for both neurons
spike_times_neuron1 = generate_poisson_spikes(r1, T)
spike_times_neuron2 = generate_poisson_spikes(r2, T)

plt.plot(spike_times_neuron2)
plt.plot(spike_times_neuron1)
# # Plot the spike times
# plt.figure(figsize=(10, 5))
# plt.eventplot(
#     [spike_times_neuron1, spike_times_neuron2], colors=["blue", "red"], linelengths=0.8
# )
# plt.title("Spike Times of Two Poisson Neurons")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Neurons")
# plt.yticks([0, 1], ["Neuron 1 (r1)", "Neuron 2 (r2)"])
# plt.xlim(0, T)
# plt.grid(True)
# plt.show()


# %%
# f) JUST FOR FUN: Extend your ROC function to take two additional input vectors weights1 and
# weights2 which specify how often a specific sample in samples1 and samples2 occur, respectively.
# This gives you the opportunity to compute the ROC for a tabulated distribution function. Re-compute
# the Poisson example for the tabulated distribution, and compare with the sampled distributions for
# different number of samples drawn.
