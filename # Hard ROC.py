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
import scipy
from sklearn.metrics import roc_curve, auc, roc_auc_score 
from sklearn.model_selection import train_test_split

# %% 
# data
# example distributions, check for errors (empty array, array input not string, list same length?, arrays the same?)
samples_t1 = "Schnubbel"        # not strings
samples_t2 = "Dubbel"

samples_a = np.array([])                # not empty
samples_b = np.array([0,])
samples_c = np.array([1,])

samples_d = np.array([0, 1, 1, 1, 1, 1])        # with roc, calculate variance, raise error if the same variance
samples_e = np.array([1, 1, 1, 1, 1, 2])

samples_f = np.array([1.001, 1.002, 1.003, 42000])      # no mixed floats & ints
samples_g = np.array([1.0021, 1.0028, 1.0029, 1.0027])

samples_h = np.array([1, 1, 1, 1, 1, 1, 1, 1])      # same length
samples_i = np.array([1, 1])

samples_1: list         # original vector 1
samples_2: list         # original vector 2
samples_1_train: list   # training data 1
samples_2_train: list   # training data 2
samples_1_test: list    # testing data 1
samples_2_test: list    # testing data 2
z: float                # decision treshold

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
# Error and excption handling
#sample_array_1 = []
#sample_array_2 = []

def check_input_format(inp1, inp2):
    if not isinstance(inp1, (list, tuple, np.ndarray)):
        raise TypeError(f"Error: inp1 must be a list, tuple, or numpy array, but got {type(inp1)}")
    
    if not isinstance(inp2, (list, tuple, np.ndarray)):
        raise TypeError(f"Error: inp2 must be a list, tuple, or numpy array, but got {type(inp2)}")
    else:
        print ("Valid, is list, tuple or array")


def check_empty(inp1, inp2):
    if isinstance(inp1, (list, tuple, np.ndarray)) and len(inp1)== 0:
        raise ValueError("inp1 cannot be empty.")
    if isinstance(inp2, (list, tuple, np.ndarray)) and len(inp2) == 0:
        raise ValueError("inp2 cannot be empty.")
    else: print("Valid, not empty")

def check_type(inp1, inp2):
    if isinstance(inp1, np.ndarray):
        if not np.issubdtype(inp1.dtype, np.number):  
            raise TypeError(f"All elements in inp1 must be numeric (int or float).")
    else:
        if not all(isinstance(item, (int, float, np.integer, np.floating)) for item in inp1):
            raise TypeError(f"All elements in inp1 must be numeric (int or float).")
    if isinstance(inp2, np.ndarray):
        if not np.issubdtype(inp2.dtype, np.number): 
            raise TypeError(f"All elements in inp2 must be numeric (int or float).")
    else:
        if not all(isinstance(item, (int, float, np.integer, np.floating)) for item in inp2):
            raise TypeError(f"All elements in inp2 must be numeric (int or float).")
    print("Valid, good numbers")

def check_len(inp1, inp2):
    if len(inp1) != len(inp2):  
        raise ValueError("Input arrays need to be the same length")
    else:
        print("Valid length") 



def check_pure_type(inp1, inp2):
    """Ensure both inputs contain only integers or only floats.
    If either contains floats or mixed types, ask the user whether to convert all to floats.
    If the user refuses, raise a TypeError.
    """

    def get_type(lst):
        """Returns 'int' if all elements are integers, 'float' if all are floats, and 'mixed' if both exist."""
        has_ints = any(isinstance(item, (int, np.integer)) for item in lst)
        has_floats = any(isinstance(item, (float, np.floating)) for item in lst)

        if has_ints and has_floats:
            return "mixed"
        elif has_floats:
            return "float"
        elif has_ints:
            return "int"
        return "unknown"  # If empty or non-numeric values

    def convert_to_floats(data):
        """Convert a list/tuple/array to all floats while preserving the original format."""
        converted = [float(item) for item in data]
        if isinstance(data, tuple):
            return tuple(converted)  # Preserve tuple format
        elif isinstance(data, np.ndarray):
            return np.array(converted, dtype=np.float64)  # Preserve NumPy array format
        return converted  # Default to list

    # Identify types
    type1 = get_type(inp1)
    type2 = get_type(inp2)

    # Check if conversion is needed
    needs_conversion = "float" in (type1, type2) or "mixed" in (type1, type2) or (type1 != type2)

    if needs_conversion:
        response = input("One or both inputs contain floats or mixed types. Please check your input. Do you really want to convert all to floats? (y/n): ").strip().lower()

        if response == "yes":
            inp1, inp2 = convert_to_floats(inp1), convert_to_floats(inp2)
            print(f" Converted both inputs to floats:\n🔹 inp1: {inp1}\n🔹 inp2: {inp2}")
        else:
            raise TypeError(f" Type mismatch: inp1 contains '{type1}', inp2 contains '{type2}'. Cannot proceed without conversion.")

    else:
        print("Valid input. No conversion needed.")

    return inp1, inp2  # Return modified values



def validate_input(inp1, inp2):
    check_input_format(inp1, inp2)
    check_empty(inp1, inp2)
    check_type(inp1, inp2) 
    check_len(inp1, inp2)
    check_pure_type(inp1, inp2)

    print("Input check complete: Computer says yes")

# %% 

test_arr1 =  np.array([1, 2, 0.5, 1, 3, 1.8])        # with roc, calculate variance, raise error if the same variance
test_arr2 = np.array([2, 2, 5, 1, 1, 2.3])

validate_input(test_arr1, test_arr2)
#check_input_format(samples_t1, samples_b)
#check_empty(samples_t1, samples_b)
#check_type(samples_t1, samples_b)
#check_len(samples_t1, samples_b)

# %%
# combine as data array
# assign array_1 with label zero, array_2 with label 1 
def split_data_labels(inp1, inp2):
    data = np.concatenate((inp1, inp2))
    labels = np.array([0] * len(inp1) + [1] * len(inp2))  #inp1 gets label 0, inp2 gets label 1
# Split into Train and Test Sets
    train_y, test_y, labels_train, labels_test = train_test_split(  # from sklearn.model_selection
        data, labels, test_size=0.33, random_state=42
        )
    return train_y, test_y, labels_train, labels_test

y_train, y_test, train_label, test_label = split_data_labels(test_arr1, test_arr2)
print (y_train, y_test, train_label, test_label)

# %% markdown

# ROC 
# a) Write a function which takes two Numpy vectors samples1 and samples2 and 
# computes the ROC curve, i.e. how false positives fp(z)
# and true positives tp(z) increase with decreasing decision threshold z 

# Your function shall return two vectors fp and tp.
# For avoiding to lose information by having to choose an arbitrary bin width,
# do not compute histograms from the samples. The challenge for you is to 
# rather compute the ROC curve directly from the sample vectors. 

### Plan:
# - choose treshold (0.5?)

# initialize prediction based on poisson distribution

# - calculate true positives (is 1, predicts 1)
# - calculate true negatives (is 0, predicts 0)
# - calculate false positives (is 0, predicts 1)
# - calculate false negatives (is 1, predicts 0)

# calculate TP rate
# calculate FP rate


#all_tprs: list[float]
#all_fprs: list[float]
#poiss_distr: list

#true_y: list
#est_y: list

# input:
# training data, training labels
# decision treshold z

# initialize decision treshold z
# calculate fp
#   store
# calculate tp
#   store
# compute 
# %%

def calcTPR(TP, FN):
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0

def FPR(FP, TN):
    FP / (FP + TN) if (FP + FN) > 0 else 0


predictprobs = np.random.poisson(4, len(data_train))

class ROC_Analysis:
    def __init__(self, data_train, labels_train):
        self.data_train = np.array(data_train)
        self.labels_train = np.array(labels_train)
        self.treshold_z = []
        self.predictsprobs = np.array(predictprobs)

    def compute_rightnwrong(self):
        # apply treshold





# def plot_roc_curve(true_y, y_prob):
#    """
#    plots the roc curve based of the probabilities
#    """

#    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
#    plt.plot(fpr, tpr)
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate') 

# compute prediction
# prediction_probs(poisson) >= treshold 
# if pred == 1 and true == 1, assign TP
# if pred == 1 and true == 0, assign FP
# if pred == 0 and true == 1, assign FN
# if pred == 0 and true == 0, assign TN

# output:
# vector false positives fp
# TP-Rate = True Positives / (True Positives + False Negatives)
# vector true positives tp
# FPR = False Positives / (False Positives + True Negatives)
# plot ROC curve




# %% 
#b) Test your functions on the examples provided in the code snippet roc samples.py. They should result
# in the plots shown on the next page. Make sure that your code detects putative inconsistencies in the
# input, e.g. wrong data types or empty sample vectors. Do some decent error catching if your assertions
# fail!

# %% 
# c) Extend your function by also computing and returning the auroc (i.e., area-under-the ROC curve). The
#auroc provides the classification accuracy given two samples, with one taken from each distribution.

# %% 
# d) Extend your function by also computing and returning acc(z). acc(z) should be the classification
#accuracy acc given one sample, drawn with equal likelihood from one of the two distributions. This
#measure depends on the chosen decision threshold z.

# classification accuracy (depending on z, test for different z)

# %%
# e) Apply your function to spike counts generated by two ’Poisson’ neurons firing with constant rates r1
#and r2, respectively, over a time interval T. Show in phase space how classification accuracy with one
#sample (acc) or two samples (auroc) depend on the firing rates. Mark the boundary where accuracy
#surpasses 95% correct.


# classification accuracy one sample -95% correct
#   with different firing rates

# classification accuracy two samples - 95% correct
# with different firing rates

# %% 
# f) JUST FOR FUN: Extend your ROC function to take two additional input vectors weights1 and
# weights2 which specify how often a specific sample in samples1 and samples2 occur, respectively.
# This gives you the opportunity to compute the ROC for a tabulated distribution function. Re-compute
# the Poisson example for the tabulated distribution, and compare with the sampled distributions for
# different number of samples drawn.