import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#
# Example Input Data
input_array_1 = np.array([1.001, 1.002, 1.003, 42000]) # No spike trials (negative class)
input_array_2 = np.array([1.0021, 1.0028, 1.0029, 1.0027])  # Spike occurred (positive class)
threshold_z = 0.3  # Decision threshold

# Prepare Data for Training
X = np.concatenate((input_array_1, input_array_2))
y = np.array([0] * len(input_array_1) + [1] * len(input_array_2))

# Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

class ROCAnalysis:
    def __init__(self, X_train, y_train, threshold_z):
        """
        Initialize the class with input arrays and threshold value.
        """
        self.X_train = np.array(X_train)  # Features (spike counts)
        self.y_train = np.array(y_train)  # Labels (0 = no spike, 1 = spike)
        self.threshold_z = threshold_z  # Decision threshold


    def fitmodel(self, y_train, labels_train):
        my_logistic_reg = LogisticRegression()
        my_logistic_reg.fit(y_train, labels_train)

    # predict probabilities
        y_pred_probs = my_logistic_reg.predict_proba(y_test)

    # keep probabilities for the positive outcome only
        probabilities_logistic_posclass = y_pred_probs[:, 1]
        return probabilities_logistic_posclass

        # Compute confusion matrix elements
    def compute_confusion_matrix(self):
        """
        Compute TP, FP, TN, FN based on threshold_z.
        """
        y_pred_class = (self.y_pred >= self.threshold_z).astype(int)  # Apply threshold

        self.TP = np.sum((y_pred_class == 1) & (self.y_train == 1))  # True Positives
        self.FP = np.sum((y_pred_class == 1) & (self.y_train == 0))  # False Positives
        self.TN = np.sum((y_pred_class == 0) & (self.y_train == 0))  # True Negatives
        self.FN = np.sum((y_pred_class == 0) & (self.y_train == 1))  # False Negatives
        return self.TP, self.FP, self.TN, self.FN

    def TPR(self):
        """Compute True Positive Rate (Recall)"""
        return self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0

    def FPR(self):
        """Compute False Positive Rate"""
        return self.FP / (self.FP + self.TN) if (self.FP + self.TN) > 0 else 0

    def predict(self, value):
        """
        Predict whether a given value comes from class 0 (no spike) or class 1 (spike).
        """
        return 1 if value >= self.threshold_z else 0  # 1 = Spike, 0 = No Spike

    def predict_batch(self, values):
        """
        Predict for a batch of values.
        """
        return np.array([self.predict(v) for v in values])

    def plot_roc_curve(self):
        """
        Generate and plot the ROC curve for various thresholds.
        """
        fpr, tpr, _ = roc_curve(self.y_train, self.y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(
            fpr, tpr, marker="o", linestyle="-", color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    def summary(self):
        """
        Print summary of the confusion matrix and performance metrics.
        """
        print(f"Threshold (z): {self.threshold_z}")
        print(f"True Positives (TP): {self.TP}")
        print(f"False Positives (FP): {self.FP}")
        print(f"True Negatives (TN): {self.TN}")
        print(f"False Negatives (FN): {self.FN}")
        print(f"True Positive Rate (TPR): {self.TPR():.2f}")
        print(f"False Positive Rate (FPR): {self.FPR():.2f}")

# Initialize ROC Analysis
roc = ROCAnalysis(X_train, y_train, threshold_z)
roc.summary()

# Compute ROC AUC
roc_fpr = roc.FPR()
roc_tpr = roc.TPR()
roc_auc = auc([0, roc_fpr, 1], [0, roc_tpr, 1])

# Plot ROC Curve
roc.plot_roc_curve()

# Select random test values from X_test and y_test
test_values = np.random.choice(X_test, size=min(3, len(X_test)), replace=False)
predictions = roc.predict_batch(test_values)

print(f"Test values: {test_values}")
print(f"Predictions: {predictions} (0 = No Spike, 1 = Spike)")

# Plot Final ROC Curve
plt.figure(figsize=(6, 6))
plt.plot([0, roc_fpr, 1], [0, roc_tpr, 1], marker='o', linestyle='-', color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Poisson-Based Predictions and Decision Threshold z')
plt.legend(loc='lower right')
plt.grid()
plt.show()
