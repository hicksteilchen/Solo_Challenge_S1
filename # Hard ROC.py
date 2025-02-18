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
samples_t1 = "Schnubbel"
samples_t2 = "Dubbel"

samples_a = np.array([])
samples_b = np.array([0,])
samples_c = np.array([1,])

samples_d = np.array([0, 1, 1, 1, 1, 1])
samples_e = np.array([1, 1, 1, 1, 1, 2])

samples_f = np.array([1.001, 1.002, 1.003, 42000])
samples_g = np.array([1.0021, 1.0028, 1.0029, 1.0027])

samples_h = np.array([1, 1, 1, 1, 1, 1, 1, 1])
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
#sample_array_1 = []
#sample_array_2 = []

def check_input_type(inp1, inp2):
    if not isinstance((inp1), (list, np.ndarray)) or not isinstance(inp2, (list, np.ndarray)): #or not isinstance((inp1, inp2), (np.ndarray)):
        raise TypeError("Input is not a list or an array")
#    elif not isinstance((inp1, inp2), (list)):  # check if list or array
        raise TypeError("Input is not an array, please give list or array")
    else:
        print ("Valid input")


#def validate_input(inp1, inp2):

#    if not (inp1, inp2):  # check if list empty
#        raise ValueError((f"Input {inp1} cannot be empty."), (f"Input {inp2} cannot be empty."))

#    if not all(isinstance(item, (int, float)) for item in inp1) or not all(isinstance((item, (int, float)) for item in inp2)): # Check all elements
#        raise TypeError("All elements must be either int or float.")

#    if not (len(inp1) == len(inp2)):
#        raise ValueError("Input arrays need to be the same length")

#    return True  # If all checks pass, return True (or proceed with function logic)

# %% 
#type(samples_i)
validate_input(samples_h, samples_i)
check_input_type(samples_h, samples_i)
# %%
# combine as data array
# assign array_1 with label zero, array_2 with label 1 

# %%


def data_ar(arr_1, arr_2):
    try:
        data = np.array(int(arr_1) + int(arr_2))
    except: ValueError
    print("Input has to be integer")

# %% 

data = np.array(sample_array_1 + sample_array_2)
labels = np.array([0] * len(sample_array_1) + [1] * len(sample_array_2))
# Split into Train and Test Sets
data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels, test_size=0.33, random_state=42
)

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
# compute ROC curve (how do fp and tp increase with decreasing z?)


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

# %%
# e) Apply your function to spike counts generated by two ’Poisson’ neurons firing with constant rates r1
#and r2, respectively, over a time interval T. Show in phase space how classification accuracy with one
#sample (acc) or two samples (auroc) depend on the firing rates. Mark the boundary where accuracy
#surpasses 95% correct.

# %% 
# f) JUST FOR FUN: Extend your ROC function to take two additional input vectors weights1 and
# weights2 which specify how often a specific sample in samples1 and samples2 occur, respectively.
# This gives you the opportunity to compute the ROC for a tabulated distribution function. Re-compute
# the Poisson example for the tabulated distribution, and compare with the sampled distributions for
# different number of samples drawn.