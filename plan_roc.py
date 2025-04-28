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


# 1. prepare data:
# label data
# Split into training and test

# initialize y_true array
# generate poisson distributed probabilities
# define treshold values z

# implement error conditions
# - no empty input arrays
# - only input arrays with ints/floats

# for each z,
#   predict y
#   calc true positives
#   calc true negatives
#   calc false positives
#   calc false negatives

#   calc TPR
#   calc FPR

# compute and return auroc

# compute and return classification accuracy

# initialize 2 spiking poisson neurons

# apply function to them


# input arrays
x = []
y = []

# split in test and training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
