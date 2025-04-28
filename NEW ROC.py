# NEW ROC
# %%
import numpy as np
import matplotlib.pyplot as plt

# %%

# Parameters
n_trials = 100  # number of trials
duration = 1.0  # one trial in s
rate_neuron1 = 20  # hz
rate_neuron2 = 40  # Hz


# %%
# simulate spike counts
def simulate_spike_counts(rate, duration, n_trials):
    spike_counts = np.random.poisson(lam=rate * duration, size=n_trials)
    return np.array(spike_counts)


# Simulate spike counts
sample_1 = simulate_spike_counts(rate_neuron1, duration, n_trials)
sample_2 = simulate_spike_counts(rate_neuron2, duration, n_trials)

# %%
# Combine samples and create corresponding labels
scores = np.concatenate([sample_1, sample_2])
labels = np.concatenate(
    [np.zeros(len(sample_1)), np.ones(len(sample_2))]
)  # 0 = from sample 1, 1 = from sample 2

# Sort scores and labels in descending order
# sorted_indices = np.argsort(-scores)
# sorted_labels = labels[sorted_indices]

# Get all unique thresholds from scores
thresholds = np.sort(np.unique(scores))[::-1]  # descending order


# %%
# TPR and FPR for each threshold
def calc_predictions(scores, labels, thresholds):
    tpr_list = []
    fpr_list = []
    accuracy_list = []
    threshold_list = []
    for thresh in thresholds:
        predictions = (scores >= thresh).astype(int)

        TP = np.sum((predictions == 1) & (labels == 1))
        FP = np.sum((predictions == 1) & (labels == 0))
        FN = np.sum((predictions == 0) & (labels == 1))
        TN = np.sum((predictions == 0) & (labels == 0))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        ACC = (TP + TN) / (TP + TN + FP + FN)

        tpr_list.append(TPR)
        fpr_list.append(FPR)
        accuracy_list.append(ACC)
        threshold_list.append(thresh)
    # Add the (0,0) and (1,1) endpoints to make the curve start and end at proper corners
    tpr_list = [0] + tpr_list + [1]
    fpr_list = [0] + fpr_list + [1]

    auc = 0.0
    for i in range(1, len(fpr_list)):
        x_diff = fpr_list[i] - fpr_list[i - 1]
        y_avg = (tpr_list[i] + tpr_list[i - 1]) / 2
        auc += x_diff * y_avg

    print(f"AUC: {auc:.4f}")
    return tpr_list, fpr_list, accuracy_list, threshold_list, auc


# %%

tpr_list, fpr_list, accuracy_list, threshold_list, auc = calc_predictions(
    scores, labels, thresholds
)

print(calc_predictions(scores, labels, thresholds)[2::2])

# %%


def calc_opt_thresh(tprs, fprs, threshs):
    # Calculate Youden's J statistic
    j_scores = np.array(tprs) - np.array(fprs)
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = threshs[optimal_idx]
    return optimal_threshold, optimal_idx


opti_thresh = calc_opt_thresh(tpr_list, fpr_list, thresholds)

# Print the optimal threshold and corresponding Youden's J
print(f"Optimal Threshold: {opti_thresh}")


# %%
acc_threshold = 0.95
marker_idx = None

for i, acc in enumerate(accuracy_list):
    if acc >= acc_threshold:
        marker_idx = i + 1  # +1 because we prepended (0,0) to the ROC lists
        break

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr_list, tpr_list, label="ROC Curve", lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1)

# Mark the first point where accuracy surpasses 95%
if marker_idx is not None:
    plt.scatter(
        fpr_list[marker_idx],
        tpr_list[marker_idx],
        color="red",
        label=f"Acc ≥ 95% @ thresh={threshold_list[marker_idx-1]}",
    )
    plt.text(
        fpr_list[marker_idx] + 0.02,
        tpr_list[marker_idx] - 0.05,
        f"Acc = {accuracy_list[marker_idx-1]*100:.1f}%",
        color="red",
    )

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve with 95% Accuracy Marker")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%
firing_rates_1 = range(20, 60)  # Firing rates for neuron 1
firing_rates_2 = range(20, 60)  # np.linspace(30, 60)  # Firing rates for neuron 2


opt_accuracies = np.zeros((len(firing_rates_1), len(firing_rates_2)))
auroc_values = np.zeros((len(firing_rates_1), len(firing_rates_2)))

thresholds = np.sort(np.unique(scores))[::-1]
# thresholds = np.linspace(0, 100, 100)  # Set thresholds based on expected spike count range

for i, r1 in enumerate(firing_rates_1):
    for j, r2 in enumerate(firing_rates_2):
        neuron_1 = simulate_spike_counts(r1, duration, n_trials)
        neuron_2 = simulate_spike_counts(r2, duration, n_trials)

        neuron_scores = np.concatenate([neuron_1, neuron_2])
        neuron_labels = np.concatenate(
            [np.zeros(len(neuron_1)), np.ones(len(neuron_2))]
        )

        # Calculate prediction stats
        tpr_list, fpr_list, accuracy_list, threshold_list, auc = calc_predictions(
            neuron_scores, neuron_labels, thresholds
        )

        # Find optimal threshold
        opti_thresh, opti_idx = calc_opt_thresh(
            tpr_list[1:-1], fpr_list[1:-1], threshold_list
        )  # skip first and last point (0,0) and (1,1)

        # Now evaluate accuracy at optimal threshold
        predictions = (neuron_scores >= opti_thresh).astype(int)
        opt_accuracy = np.mean(predictions == neuron_labels)

        # Save results
        opt_accuracies[i, j] = opt_accuracy
        auroc_values[i, j] = auc

print("Finished computing accuracies and AUROCs")

# %%
# Plot the results in phase space
plt.figure(figsize=(12, 6))

# Accuracy contour plot
plt.subplot(1, 2, 1)
plt.contourf(
    firing_rates_1,
    firing_rates_2,
    opt_accuracies,
    levels=np.linspace(0, 1, 21),
    cmap="viridis",
    alpha=0.7,
)
plt.colorbar(label="Accuracy")
plt.contour(
    firing_rates_1,
    firing_rates_2,
    opt_accuracies,
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
    firing_rates_1,
    firing_rates_2,
    auroc_values,
    levels=np.linspace(0, 1, 21),
    cmap="viridis",
    alpha=0.7,
)
plt.colorbar(label="AUC")
plt.title("AUC Phase Space")
plt.xlabel("Firing Rate of Neuron 1 (r1)")
plt.ylabel("Firing Rate of Neuron 2 (r2)")
plt.contour(
    firing_rates_1,
    firing_rates_2,
    auroc_values,
    levels=[0.95],
    colors="red",
    linewidths=2,
    linestyles="dashed",
)

plt.tight_layout()
plt.show()

# %%
# x = np.concatenate([sample_1, sample_2])  # neuron 1
# y = np.concatenate([sample_2, sample_1])  # neuron 2
# colors = ['blue' if l == 0 else 'orange' for l in labels]

# # Plot
# plt.clf
# plt.figure(figsize=(7, 6))
# plt.scatter(x, y, c=colors, alpha=0.6, label='Trials')
# plt.xlabel('Neuron 1 Spike Count')
# plt.ylabel('Neuron 2 Spike Count')
# plt.title('Phase Space with Decision Boundary')
# plt.grid(True)

# # Plot decision boundary: y = x + best_thresh
# x_vals = np.linspace(min(x), max(x), 100)
# y_vals = x_vals + optimal_threshold
# plt.plot(x_vals, y_vals, 'r--', linewidth=2)

# plt.legend()
# plt.tight_layout()
# plt.show()

