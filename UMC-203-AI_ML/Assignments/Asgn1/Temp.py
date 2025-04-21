# Code Block: 1

import numpy as np
import matplotlib.pyplot as plt
from oracle import q2_train_test_emnist

# Load the dataset
data = q2_train_test_emnist(23684, "../EMNIST/emnist-balanced-train.csv", "../EMNIST/emnist-balanced-test.csv")

train_data = data[0]
test_data = data[1]

# Changing the labels to 0 and 1 from 44 and 12 respectively

# Code Block: 2

for idx in range(len(train_data)):
    if train_data[idx][0] == 12:
        train_data[idx][0] = 1
    else:
        train_data[idx][0] = 0

for idx in range(len(test_data)):
    if test_data[idx][0] == 12:
        test_data[idx][0] = 1
    else:
        test_data[idx][0] = 0

# Separating the data into class 0 and class 1

class0_data = []
class1_data = []

for data in train_data:
    if data[0] == 0:
        class0_data.append(data)
    else:
        class1_data.append(data)


# Code Block: 3

# Different class distributions
splits = {
    "50-50": (len(class0_data), len(class1_data)),
    "60-40": (int(0.6 * len(train_data)), int(0.4 * len(train_data))),
    "80-20": (int(0.8 * len(train_data)), int(0.2 * len(train_data))),
    "90-10": (int(0.9 * len(train_data)), int(0.1 * len(train_data))),
    "99-1": (int(0.99 * len(train_data)), int(0.01 * len(train_data))),
}

# Epsilon values
epsilon_values = [0.01,0.1, 0.25, 0.4]

results = {}

# Code Block: 4

def Gaussian_Distribution(x, mean, cov_inv, det):
    d = len(mean)
    dx = x - mean
    pow = -0.5 * np.dot(np.dot(dx, cov_inv), dx)
    return pow - 0.5 * det - (d / 2) * np.log(2 * np.pi)

# Code Block: 5

def Modified_Bayes_Classifier(test_data, mean_class0, mean_class1, cov_inv0, cov_inv1, 
                             det0, det1, p0, p1, epsilon):
    """ Classifies test samples using the Modified Bayes Classifier """
    
    predictions = []
    
    for x in test_data:
        log_p0 = Gaussian_Distribution(x, mean_class0, cov_inv0, det0) + np.log(p0)
        log_p1 = Gaussian_Distribution(x, mean_class1, cov_inv1, det1) + np.log(p1)

        max_log = max(log_p0, log_p1)
        denominator = np.exp(log_p0 - max_log) + np.exp(log_p1 - max_log)
        
        if denominator == 0:
            eta_x = 0.5  # Assign equal probability if denominator is zero
        else:
            eta_x = np.exp(log_p1 - max_log) / denominator

        if eta_x >= 0.5 + epsilon:
            predictions.append(1)  # Class 1
        elif eta_x <= 0.5 - epsilon:
            predictions.append(0)  # Class 0
        else:
            predictions.append(-1)  # Reject

    return np.array(predictions)


# Code Block: 6


import numpy as np

def Testing(test_data, test_labels, mean_class0, mean_class1, cov_inv0, cov_inv1, det0, det1, prior_prob_class0, prior_prob_class1, epsilon):
    
    # Get predictions from the Modified Bayes Classifier
    result = Modified_Bayes_Classifier(test_data, mean_class0, mean_class1, cov_inv0, cov_inv1, det0, det1, prior_prob_class0, prior_prob_class1, epsilon)

    # Store non-rejected samples
    non_rejected_pred = []
    non_rejected_labels = []

    # Iterate through predictions and filter out rejected samples
    for i in range(len(result)):
        if result[i] != -1:  # If not rejected
            non_rejected_pred.append(result[i])
            non_rejected_labels.append(test_labels[i])

    # Convert lists to NumPy arrays
    non_rejected_pred = np.array(non_rejected_pred)
    non_rejected_labels = np.array(non_rejected_labels)

    # Compute misclassification loss
    bad_pred = 0
    total_accepted = len(non_rejected_pred)

    for i in range(total_accepted):
        if non_rejected_pred[i] != non_rejected_labels[i]:
            bad_pred += 1

    if total_accepted > 0:
        misclassification_loss = bad_pred / total_accepted
    else:
        misclassification_loss = 0  # Avoid division by zero

    # Compute rejection count
    rejected_cnt = 0
    for i in range(len(result)):
        if result[i] == -1:
            rejected_cnt += 1

    rejection_rate = rejected_cnt / len(result)
    print(f"for epsilon = {epsilon}\n")
    print("Rejection rate: ", rejection_rate)
    print("Misclassification loss: ", misclassification_loss)
    print("Rejected count: ", rejected_cnt, "\n")
    return misclassification_loss, rejected_cnt

# Code Block: 7

def Classifier(train_data, test_data_feat, test_labels, epsilons):
    """Train and test classifier for given dataset"""
    
    # Separate class data
    class0_train = train_data[train_data[:, 0] == 0][:, 1:]
    class1_train = train_data[train_data[:, 0] == 1][:, 1:]

    # prior probabilities
    p0 = len(class0_train) / len(train_data)
    p1 = len(class1_train) / len(train_data)

    # means
    mean_class0 = np.mean(class0_train, axis=0)
    mean_class1 = np.mean(class1_train, axis=0)

    # covariance matrices
    cov_class0 = np.cov(class0_train, rowvar=False)
    cov_class1 = np.cov(class1_train, rowvar=False)

    # Since the covariance matrices are singular, i am adding a small value to the diagonal matrix to them
    cov_class0 += (1e-9) * np.identity(cov_class0.shape[0])
    cov_class1 += (1e-9) * np.identity(cov_class1.shape[0])

    # pseudo-inverses
    cov_inv0 = np.linalg.pinv(cov_class0)
    cov_inv1 = np.linalg.pinv(cov_class1)

    # log determinants
    det0 = np.linalg.slogdet(cov_class0)[1]
    det1 = np.linalg.slogdet(cov_class1)[1]

    misclassification_losses = []
    num_rejected = []

    for epsilon in epsilons:
        loss, rejected = Testing(test_data_feat, test_labels, mean_class0, mean_class1, cov_inv0, cov_inv1, det0, det1, p0, p1, epsilon)
        misclassification_losses.append(loss)
        num_rejected.append(rejected)

    return misclassification_losses, num_rejected

# Code Block: 8

# Extract test labels and features
test_labels = test_data[:, 0]
test_data_feat = test_data[:, 1:]

# Compute results for each prior split

for split, (num_class0, num_class1) in splits.items():
    print(f"\nProcessing split: {split} \n")
    sub_class0 = class0_data[:num_class0]
    sub_class1 = class1_data[:num_class1]
    modified_train_data = np.vstack((sub_class0, sub_class1))

    losses, rejected = Classifier(modified_train_data, test_data_feat, test_labels, epsilon_values)
    
    results[split] = {"losses": losses, "rejected": rejected}

plt.figure(figsize=(8, 5))

for split in results.keys():
    plt.plot(epsilon_values, results[split]["losses"], marker='o', linestyle='-', label=split)

plt.xlabel("Epsilon (ε)")
plt.ylabel("Misclassification Loss")
plt.title("Misclassification Loss vs. Epsilon for Different Class Priors")
plt.legend()
plt.grid(True)
plt.show()

# Code Block: 9

for split in results.keys():
    print(f"Class Distribution: {split}")
    print(f"Losses: {results[split]['losses']}")
    print(f"Rejected: {results[split]['rejected']}")
    print()
    
