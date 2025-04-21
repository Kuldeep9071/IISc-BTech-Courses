import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# Importing the functions from oracle
from oracle import q1_fish_train_test_data
from oracle import q2_train_test_emnist
from oracle import q3_hyper


"""Question 1 : Fisher's Linear Discriminant"""


data = q1_fish_train_test_data(23684)

my_attributes = data[0] # tuple with 2 elements
train_img = np.array(data[1])
train_label = np.array(data[2])
test_img = np.array(data[3])
test_label = np.array(data[4])

print(my_attributes)
train_img = train_img*255
test_img =  test_img*255

n = len(train_img) # 20000 number of training images
x = len(train_img[0]) # 3 
y = len(train_img[0][0]) # 32
z = len(train_img[0][0][0]) # 32

# Converting each image tensor into a vector of dimension 1 x (3*32*32) = 1 x 3072

train_img = train_img.reshape(20000, -1)
test_img = test_img.reshape(1000, -1)


def Classify_data(train_img, train_label, num_sample):
    """Input : Train_data, Train_label, num_sample
       Output : 4 classes of data after taking random samples of size num_sample from the train data"""
    
    sub_idx = np.random.choice(len(train_img), num_sample, replace=False)
    class0, class1, class2, class3 = [], [], [], []

    for i in range(num_sample):
        if train_label[sub_idx[i]] == 1:
            class1.append(train_img[sub_idx[i]])
        elif train_label[sub_idx[i]] == 2:
            class2.append(train_img[sub_idx[i]])
        elif train_label[sub_idx[i]] == 3:
            class3.append(train_img[sub_idx[i]])
        else:
            class0.append(train_img[sub_idx[i]])
    
    return np.array(class0), np.array(class1), np.array(class2), np.array(class3)


# Calculating the mean and varienve of each class
def select_random_indices(n, num_sample):
    return np.random.choice(n, num_sample, replace=False)

num_sample_arr = [50, 100, 500, 1000, 2000, 4000]

L2_norm_means = []
Fro_norm_means = []

for num_sample in num_sample_arr:
    class0, class1, class2, class3 = Classify_data(train_img, train_label, num_sample)

    mean0 = np.mean(class0, axis=0)
    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)
    mean3 = np.mean(class3, axis=0)

    # Calculating L2 norm of mean and frobenius norm of variance

    norm0 = np.linalg.norm(mean0, ord=2)
    norm1 = np.linalg.norm(mean1, ord=2)
    norm2 = np.linalg.norm(mean2, ord=2)
    norm3 = np.linalg.norm(mean3, ord=2)

    var0 = np.cov(class0, rowvar=False)
    var1 = np.cov(class1, rowvar=False)
    var2 = np.cov(class2, rowvar=False)
    var3 = np.cov(class3, rowvar=False)

    frob_norm0 = np.linalg.norm(var0, ord='fro')
    frob_norm1 = np.linalg.norm(var1, ord='fro')
    frob_norm2 = np.linalg.norm(var2, ord='fro')
    frob_norm3 = np.linalg.norm(var3, ord='fro')

    # Appending the values to the mean and varience list

    L2_norm_means.append([norm0, norm1, norm2, norm3])
    Fro_norm_means.append([frob_norm0, frob_norm1, frob_norm2, frob_norm3])


# Plotting the graph

# L2 norm of means vs Number of samples

plt.figure(figsize=(7,4))
plt.plot(num_sample_arr, L2_norm_means, marker='o')
plt.xlabel("Number of samples")
plt.ylabel("L2 Norm of means")
plt.title("L2 Norm of means vs Number of samples")
plt.legend(["Class 0", "Class 1", "Class 2", "Class 3"])
plt.show()

# Frobenius norm of Co-Variance matrix vs Number of samples

plt.figure(figsize=(7,4))
plt.plot(num_sample_arr, Fro_norm_means, marker='o')
plt.xlabel("Number of samples")
plt.ylabel("Frobenius Norm of variance")
plt.title("Frobenius Norm of variance vs Number of samples")
plt.legend(["Class 0", "Class 1", "Class 2", "Class 3"])
plt.show()


def My_FLD(X, y, n_classes=4):
    """Input : train_img, train_label, n_classes
       Output : weights and objective value of FLD"""
    class_labels = np.unique(y)
    mean_vectors = {c: np.mean(X[y == c], axis=0) for c in class_labels}
    
    # Compute within-class scatter matrix S_W
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for c in class_labels:
        
        class_samples = X[y == c]  
        class_cov = np.cov(class_samples, rowvar=False, bias=True)
        class_scatter = class_cov * (class_samples.shape[0] - 1)

        S_W += class_scatter

    
    # Compute between-class scatter matrix S_B
    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((X.shape[1], X.shape[1]))
    for c in class_labels:
        n_c = X[y == c].shape[0]
        mean_diff = (mean_vectors[c] - overall_mean).reshape(-1, 1)
        S_B += n_c * (mean_diff @ mean_diff.T)

    # Solve the generalized eigenvalue problem S_B w = λ S_W w
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)

    # Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigvals)[::-1]
    fld_weights = eigvecs[:, sorted_indices[:n_classes - 1]] # Choose n_classes - 1 eigenvectors

    # Compute objective function value (trace of S_B / S_W)
    Obj_Val = np.trace(np.linalg.pinv(S_W) @ S_B)

    return fld_weights, Obj_Val


def select_random_indices(total_size, num_sample):
    """Selects random indices without replacement."""
    return np.random.choice(total_size, num_sample, replace=False)


sample_sizes = [2500, 3500, 4000, 4500, 5000]
fld_weights = [] 

for n in sample_sizes:
    if n == 5000:
        # Train on a single full dataset
        indices = select_random_indices(len(train_img), n)
        subset_img, subset_label = train_img[indices], train_label[indices]
        weight_fld = My_FLD(subset_img, subset_label)
        fld_weights.append((weight_fld[0], weight_fld[1]))
    else:
        temp_weights = []  # Store 20 subsets for each sample size
        for subset in range(20):
            indices = select_random_indices(len(train_img), n)
            subset_img, subset_label = train_img[indices], train_label[indices]
            weight_fld = My_FLD(subset_img, subset_label)
            temp_weights.append((weight_fld[0], weight_fld[1]))
        fld_weights.append(temp_weights)

print("FLD training completed for all sample sizes!")

# Print FLD weights and objective function values
for i, n in enumerate(sample_sizes):
    if n == 5000:
        print(f"Weight for sample size {n}: {fld_weights[i][0]}")
        print(f"Objective function value for sample size {n}: {fld_weights[i][1]}")
    else:
        for j in range(20):
            print(f"Weight for sample size {n}, subset {j}: {fld_weights[i][j][0]}")
            print(f"Objective function value for sample size {n}, subset {j}: {fld_weights[i][j][1]}")

# Box plot for multi-class objective function values
box_plot_data = [
    [fld_weights[i][j][1] for j in range(20)] if sample_sizes[i] != 5000 else [fld_weights[i][1]]
    for i in range(len(sample_sizes))
]

plt.figure(figsize=(8, 5))
plt.boxplot(box_plot_data, labels=[str(n) for n in sample_sizes], patch_artist=True)
plt.xlabel("Number of Samples (n)")
plt.ylabel("Objective Function Value")
plt.title("Box Plot of Multi-Class Objective Function Values")
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

W, Obj_Val = My_FLD(train_img, train_label, 5000) # Projectng 2500 samples

train_img_fld = train_img @ W  # Projecting data onto FLD space

# Plotting in 3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


for i in np.unique(train_label):
    class_indices = (train_label == i)
    ax.scatter(
        train_img_fld[class_indices, 0], 
        train_img_fld[class_indices, 1], 
        train_img_fld[class_indices, 2],
        label=f'Class {i}')


ax.set_xlabel("FLD Component 1")
ax.set_ylabel("FLD Component 2")
ax.set_zlabel("FLD Component 3")
ax.set_title("3D Projection of Data using Fisher's Linear Discriminant")
ax.legend()
plt.show()


""" Question 2: Bayes Classification """


# Load the dataset
data = q2_train_test_emnist(23684, "../EMNIST/emnist-balanced-train.csv", "../EMNIST/emnist-balanced-test.csv")

train_data = data[0]
test_data = data[1]


# Changing the labels to 0 and 1 from 44 and 12 respectively

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

def Gaussian_Distribution(x, mean, cov_inv, det):
    d = len(mean)
    dx = x - mean
    pow = -0.5 * np.dot(np.dot(dx, cov_inv), dx)
    return pow - 0.5 * det - (d / 2) * np.log(2 * np.pi)

def Modified_Bayes_Classifier(test_data, mean_class0, mean_class1, cov_inv0, cov_inv1, det0, det1, p0, p1, epsilon):
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

for split in results.keys():
    print(f"Class Distribution: {split}")
    print(f"Losses: {results[split]['losses']}")
    print(f"Rejected: {results[split]['rejected']}")
    print()
    

# Number of folds
K = 5
epsilon = 0.25

kf = KFold(n_splits=K, shuffle=True, random_state=37)

# Store performance metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
misclassification_losses = []
rejection_rates = []

def Performence_Calculator(TP, TN, FP, FN, rejected_samples, total_samples):
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    misclassification_loss = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    rejection_rate = rejected_samples / total_samples if total_samples > 0 else 0
    return accuracy, precision, recall, f1_score, misclassification_loss, rejection_rate


X = train_data[:, 1:]
y = train_data[:, 0]

# Perform 5-Fold Cross Validation
for train_index, val_index in kf.split(X):
    # Split into training and validation sets
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Compute class-wise statistics
    mean_class0 = np.mean(X_train[y_train == 0], axis=0)
    mean_class1 = np.mean(X_train[y_train == 1], axis=0)

    cov_class0 = np.cov(X_train[y_train == 0], rowvar=False) + (1e-9) * np.identity(X_train.shape[1])
    cov_class1 = np.cov(X_train[y_train == 1], rowvar=False) + (1e-9) * np.identity(X_train.shape[1])

    cov_inv0 = np.linalg.pinv(cov_class0)
    cov_inv1 = np.linalg.pinv(cov_class1)

    det0 = np.linalg.slogdet(cov_class0)[1]
    det1 = np.linalg.slogdet(cov_class1)[1]

    p0 = np.sum(y_train == 0) / len(y_train)
    p1 = np.sum(y_train == 1) / len(y_train)

    predictions = Modified_Bayes_Classifier(X_val, mean_class0, mean_class1, cov_inv0, cov_inv1, det0, det1, p0, p1, epsilon)

    valid_indices = predictions != -1
    cm = confusion_matrix(y_val[valid_indices], predictions[valid_indices], labels=[0, 1])

    # Extract TP, TN, FP, FN
    TP = cm[1, 1]  # True Positives
    TN = cm[0, 0]  # True Negatives
    FP = cm[0, 1]  # False Positives
    FN = cm[1, 0]  # False Negatives

    # Compute rejected samples
    rejected_samples = np.sum(predictions == -1)
    total_samples = len(y_val)

    # Compute performance metrics
    accuracy, precision, recall, f1, misclassification_loss, rejection_rate = Performence_Calculator(TP, TN, FP, FN, rejected_samples, total_samples)

    # Store metrics

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    misclassification_losses.append(misclassification_loss)
    rejection_rates.append(rejection_rate)

# Compute average metrics over all folds
accuracy = np.mean(accuracy_scores)
precision = np.mean(precision_scores)
recall = np.mean(recall_scores)
f1 = np.mean(f1_scores)
misclassification_loss = np.mean(misclassification_losses)
rejection_rate = np.mean(rejection_rates)
CM = np.array([[TN, FP], [FN, TP]]) # Confusion Matrix

# Print Final Performance

print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Accuracy: {accuracy}")
print(f"F1-score: {f1}")


print("Number of Rejected Samples: ", rejection_rate)
print(f"Misclassification Loss: {misclassification_loss}")

print("Confusion Matrix: \n", CM)



""" Question 3: Decision Tree Classifier """

# Reading and Cleaning the data

df = pd.read_csv("../heart+disease/processed.cleveland.data") 

df.replace("?",np.nan,inplace=True)
df.dropna(inplace=True)

X = df.iloc[:, :-1] # first 13 columns are features
Y = df.iloc[:, -1]  # 14th (last) column is target

""" Since we have to classify the target into 2 classes, we will replace all values greater than 0 to 1 because
        positive values indicates heart disease and the magnitue the intensity of the disease"""

for target in Y:
    if target > 0:
        Y.replace(target, 1, inplace=True) 


# Splitting the data into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=37) # 80% training and 20% testing


# Getting the hyperparameters from the oracle using my SR Number

MyInput = q3_hyper(23684)
print("My Input from orcale: " , MyInput) # printing my input

My_tree_model = DecisionTreeClassifier(criterion=MyInput[0] , splitter=MyInput[1]  , max_depth = MyInput[2])

My_tree_model.fit(X_train, Y_train)

plt.figure(figsize=(15,10))

tree.plot_tree(My_tree_model,filled=True)

# How good is this Disicion Tree Classifier : Analysis

Y_prediciton = My_tree_model.predict(X_test) # Predicting the target values

TP = 0 # True Positive
FP = 0  # False Positive  
FN = 0  # False Negative
TN= 0   # True Negative

for actual, predicted in zip(Y_test, Y_prediciton):
    if actual == 1 and predicted == 1:
        TP += 1
    elif actual == 0 and predicted == 1:
        FP += 1
    elif actual == 1 and predicted == 0:
        FN += 1
    else:
        TN += 1

# Calculating metrics

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
F1_Score = 2 * (Precision * Recall) / (Precision + Recall)

# Printing the metrics
print("Precision of the model:", Precision)
print("Recall of the model:", Recall)
print("F1 Score of the model:", F1_Score)
print("Accuracy of the model:", Accuracy)