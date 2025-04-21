import os
import cv2
import time
import cvxopt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# Oracle functions

from Oracle_Assignment_2 import q1_get_cifar100_train_test
from Oracle_Assignment_2 import q2_get_mnist_jpg_subset
from Oracle_Assignment_2 import q3_linear_1
from Oracle_Assignment_2 import q3_linear_2
from Oracle_Assignment_2 import q3_stocknet

""" Question 1: Support Vector Machine (SVM) """

# Load Data

q1_data = q1_get_cifar100_train_test(23684)

train_data = q1_data[0]
test_data = q1_data[1]

train_features = np.array([x[0] for x in train_data])
train_labels = np.array([x[1] for x in train_data]) 

test_features = np.array([x[0] for x in test_data])
test_labels = np.array([x[1] for x in test_data]) 

# Perceptrion Algorithm

total = train_features.shape[0]

def misses(features,labels,w,b):
    misses = 0
    for i in range(features.shape[0]):
        if labels[i]*(np.dot(w,features[i])+b) <= 0:
            misses += 1
    return misses

def perceptron_algorithm(features, labels, epochs):
    n_samples, n_features = features.shape
    w = np.zeros(n_features)
    b = 0
    incorrect = []
    
    for epoch in range(epochs):
        errors = 0
        for i in range(n_samples):
            if labels[i] * (np.dot(w, features[i]) + b) <= 0:
                w += labels[i] * features[i]
                b += labels[i]
                errors += 1
        incorrect.append(errors / n_samples)
    return incorrect,w, b


# it does not converged i waited around 25-30 minutes runnig for 1000 iterations


incorrect,w,b = perceptron_algorithm(train_features, train_labels,100000) 
print("w =",w)
print("b =",b)

print("Test Misses",misses(test_features,test_labels,w,b),"out of",test_features.shape[0])

# plot of missclassification rate vs iteration

plt.plot(range(100000),incorrect)
plt.xlabel('Iteration')
plt.ylabel('Missclassification Rate')
plt.title('Missclassification Rate vs Iteration')
plt.ylim(0.3,0.5)
plt.show()

# Linear SVM and Primal+Dual

# Normalizing data
train_features = (train_features - np.mean(train_features, axis=0)) / np.std(train_features, axis=0)
test_features = (test_features - np.mean(test_features, axis=0)) / np.std(test_features, axis=0)

# Primal

def Primal_SVM(X, y, C=1.0):
    data_size, num_feature = X.shape
    y = y.reshape(-1, 1)

    # Objective function
    n = num_feature + 1 + data_size 
    P = np.zeros((n, n))
    P[:num_feature, :num_feature] = np.eye(num_feature)
    q = np.hstack([np.zeros(num_feature + 1), C * np.ones(data_size)])

    # inequalities constraints
    inq1 = np.hstack([-y * X, -y, -np.eye(data_size)])
    inq2 = np.hstack([np.zeros((data_size, num_feature + 1)), -np.eye(data_size)])
    G = np.vstack([inq1, inq2])
    h = np.hstack([-np.ones(data_size), np.zeros(data_size)])

    # using cvxopt for optimization 

    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    sol = solvers.qp(P, q, G, h)
    solution = np.array(sol['x']).flatten()

    w = solution[:num_feature]
    b = solution[num_feature]
    xi = solution[num_feature + 1:]

    return w, b, xi


# Dual

def Dual_SVM(X, y, C=1.0):
    data_size, num_feature = X.shape
    y = y.reshape(-1, 1)
    
    # Objective function
    K = X @ X.T
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(data_size))
    
    # Inequality constraints
    G = np.vstack([-np.eye(data_size), np.eye(data_size)])
    h = np.hstack([np.zeros(data_size), C * np.ones(data_size)])
    
    # Equality constraints
    A = matrix(y.reshape(1, -1).astype(float))
    b = matrix(0.0)
    
    # using cvxopt for optimization
    G = matrix(G)
    h = matrix(h)

    sol = solvers.qp(P, q, G, h, A, b)    
    alpha = np.array(sol['x']).flatten()
    w = np.dot((alpha * y.flatten()).T, X)
    
    sv_indices = np.where(alpha > 1e-5)[0]
    b = np.mean(y[sv_indices] - np.dot(X[sv_indices], w))
    
    return w, b, alpha

def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)


def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

start_time_primal = time.time()

w_primal, b_primal, soln = Primal_SVM(train_features, train_labels, 1.0)
predictions_primal = predict(train_features, w_primal, b_primal)
misclassified_primal = np.where(predictions_primal != train_labels)[0]

end_time_primal = time.time()

start_time_dual = time.time()

w_dual, b_dual, soln = Dual_SVM(train_features, train_labels, 1.0)
predictions_dual = predict(train_features, w_dual, b_dual)
misclassified_dual = np.where(predictions_dual != train_labels)[0]

end_time_dual = time.time()

print("misclassified points primal =", len(misclassified_primal))
print("misclassified points dual =", len(misclassified_dual))
print("time taken for primal SVM =", end_time_primal - start_time_primal)
print("time taken for dual SVM =", end_time_dual - start_time_dual)

np.savetxt('inseparable_23684.csv', misclassified_primal, delimiter=',') # csv file

# Kernel SVM

def gaussian_kernel(X1, X2, gamma):
    return np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2)**2)

def Kernelized_SVM(X, y, C=1.0, gamma=0.1):
    m = X.shape[0]
    y = y.flatten()

    # kernel function
    K = gaussian_kernel(X, X, gamma)

    P = matrix(np.outer(y, y) * K, tc='d')
    q = matrix(-np.ones(m), tc='d')
    G = matrix(np.vstack([-np.eye(m), np.eye(m)]), tc='d')
    h = matrix(np.hstack([np.zeros(m), C * np.ones(m)]), tc='d')
    A = matrix(y.reshape(1, -1), tc='d')
    b = matrix(0.0, tc='d')

    # using cvxopt for optimization
    soln = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(soln['x']).flatten()
    
    mask = alpha > 1e-5
    vectors = X[mask]
    alphas = alpha[mask]
    labels = y[mask]
    
    margin = (alpha > 1e-5) & (alpha < C)
    if np.any(margin):
        K_sv = gaussian_kernel(X[margin], X, gamma)
        b = np.mean(y[margin] - np.dot(K_sv, alpha * y))
    else:
        K_sv = gaussian_kernel(vectors, X, gamma)
        b = np.mean(labels - np.dot(K_sv, alpha * y))
    
    return vectors, alphas, labels, b, gamma

def kernel_predict(X_test, vectors, alphas, labels, b, gamma):
    K_test = gaussian_kernel(X_test, vectors, gamma)
    decision = np.dot(K_test, alphas * labels) + b
    return np.sign(decision)

vectors, alphas, labels, b, gamma = Kernelized_SVM(train_features, train_labels, 1.0, 0.1)
predictions_train = kernel_predict(train_features, vectors, alphas, labels, b, gamma)
misclassified_train = np.where(predictions_train != train_labels)[0]

vectors, alphas, labels, b, gamma = Kernelized_SVM(test_features, test_labels, 1.0, 0.1)
predictions_test = kernel_predict(test_features, vectors, alphas, labels, b, gamma)
misclassified_test = np.where(predictions_test != test_labels)[0]

print("misclassified points in train data=", len(misclassified_train))
print("misclassified points in test data=", len(misclassified_test))

new_train_features = []
new_train_labels = []
for i in range(len(train_features)):
    if i not in misclassified_primal:
        new_train_features.append(train_features[i])
        new_train_labels.append(train_labels[i])

new_train_features = np.array(new_train_features)
new_train_labels = np.array(new_train_labels)

new_incorrect, w, b = perceptron_algorithm(new_train_features, new_train_labels, 1000)


def misses(features, labels, w, b):
    return np.sum(labels * (np.dot(features, w) + b) <= 0)

# Plot new training curve
plt.plot(range(1000), new_incorrect)
plt.xlabel('Iteration')
plt.ylabel('Misclassification Rate')
plt.title('Misclassification Rate vs Iteration (Filtered Data)')
plt.show()


""" Question 2: Logistic Regression, MLP, CNN & PCA """

Q2_data = q2_get_mnist_jpg_subset(23684)


# converting img to numpy array
data_folder = "q2_data"
data = [] 

for i in range(10):
    folder_path = os.path.join(data_folder, str(i))
    images = []

    for img_name in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_name)

        if img_name.endswith(".jpg"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)

    data.append(np.array(images))

data = np.array(data)

# MLP

images = data.reshape(-1, 28, 28)
labels = np.repeat(np.arange(10), 1000)

# Split 
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = torch.from_numpy(X_train).float().view(-1, 28*28)
X_test = torch.from_numpy(X_test).float().view(-1, 28*28)
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

# DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


Neural_Network = nn.Sequential(
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

criterion = nn.CrossEntropyLoss() # Cross-entropy loss
optimizer = torch.optim.Adam(Neural_Network.parameters(), lr=0.001) 


epochs = 20
for epoch in range(epochs):
    Neural_Network.train()
    curr_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = Neural_Network(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        curr_loss += loss.item() * images.size(0)
    
    epoch_loss = curr_loss / len(train_loader.dataset)

# Evaluation
Neural_Network.eval()
correct = 0
total = 0

for images, labels in test_loader:
    outputs = Neural_Network(images)
    probabilities = F.softmax(outputs, dim=1)
    temp, predicted = torch.max(probabilities, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print("Accuracy of the model on the test set:", (correct / total))


# CNN
X_train = X_train.view(-1, 1, 28, 28)
X_test = X_test.view(-1, 1, 28, 28)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


cnn_model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    cnn_model.train()
    curr_loss = 0.0
    for images, labels in train_loader:
        images = images.view(-1, 1, 28, 28)
        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        curr_loss += loss.item() * images.size(0)
    
    epoch_loss = curr_loss / len(train_loader.dataset)


cnn_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 1, 28, 28)
        outputs = cnn_model(images)
        probabilities = F.softmax(outputs, dim=1)
        temp, predicted = torch.max(probabilities, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Accuracy of the CNN model on the test set:", (correct / total))

# PCA
images_flat = data.reshape(data.shape[0] * data.shape[1], -1)
labels_flat = np.repeat(np.arange(10), 1000)

n_components = 50 
pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(images_flat)


X_train, X_test, y_train, y_test = train_test_split(pca_features, labels_flat, test_size=0.2, random_state=42, stratify=labels_flat)


X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()


batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


plt.figure(figsize=(10, 5))
for i in range(min(8, n_components)):
    plt.subplot(1, 8, i+1)
    plt.imshow(pca.components_[i].reshape(28, 28), cmap='gray')
    plt.title(f'PC {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# MLP with PCA
class MLP_With_PCA(nn.Module):
    def __init__(self, input_size=50):
        super(MLP_With_PCA, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create MLP model with PCA features
mlp_pca_model = MLP_With_PCA(input_size=n_components)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp_pca_model.parameters(), lr=0.001)

# Training
epochs = 30
for epoch in range(epochs):
    mlp_pca_model.train()
    curr_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = mlp_pca_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        curr_loss += loss.item() * images.size(0)
    
    epoch_loss = curr_loss / len(train_loader.dataset)


mlp_pca_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = mlp_pca_model(images)
        probabilities = F.softmax(outputs, dim=1)
        temp, predicted = torch.max(probabilities, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy of the MLP model with PCA features on the test set:", (correct / total))

# Logistic Regression with PCA
class Logistic_Regression_with_PCA(nn.Module):
    def __init__(self, input_size=50, num_classes=10):
        super(Logistic_Regression_with_PCA, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


class one_vs_rest:
    def __init__(self, input_size=50):
        self.classifiers = [nn.Linear(input_size, 1) for _ in range(10)]
        self.optimizers = [torch.optim.Adam(c.parameters(), lr=0.01) for c in self.classifiers]
        self.criterion = nn.BCEWithLogitsLoss() # binary cross-entropy loss

    def train(self, X_train, y_train, epochs=20):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(10):
                binary_labels = (y_train == i).float()
                self.optimizers[i].zero_grad()
                outputs = self.classifiers[i](X_train).squeeze()
                
                loss = self.criterion(outputs, binary_labels)
                
                loss.backward()
                self.optimizers[i].step()
                
                total_loss += loss.item()

    def predict(self, X_test):
        predictions = torch.stack([torch.sigmoid(c(X_test)).squeeze() for c in self.classifiers], dim=1)
        return torch.argmax(predictions, dim=1)

# Multi-class Logistic Regression
logistic_model = Logistic_Regression_with_PCA(input_size=n_components)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(logistic_model.parameters(), lr=0.001)

epochs = 30
for epoch in range(epochs):
    logistic_model.train()
    curr_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = logistic_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        curr_loss += loss.item() * images.size(0)
    
    epoch_loss = curr_loss / len(train_loader.dataset)


logistic_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = logistic_model(images)
        probabilities = F.softmax(outputs, dim=1)
        temp, predicted = torch.max(probabilities, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy of the Logistic Regression model with PCA features on the test set:", (correct / total))


ovr_classifier = one_vs_rest(input_size=n_components)
ovr_classifier.train(X_train, y_train)

# One-vs-Rest Evaluation
predictions = ovr_classifier.predict(X_test)
correct = (predictions == y_test).sum().item()

total = y_test.size(0)

print("One vs Rest Classifier Test Accuracy:", (correct / total))

# Image reconstruction
digit_8_images = data[8]

digit_8_flat = digit_8_images.reshape(digit_8_images.shape[0], -1)

n_components_list = [1, 10, 25, 50, 100,200,500]
plt.figure(figsize=(15, 3))


plt.subplot(1, len(n_components_list) + 1, 1)
plt.imshow(digit_8_images[0], cmap='gray')
plt.title('Original')
plt.axis('off')

# construction
for i, n_components in enumerate(n_components_list, 1):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(digit_8_flat)
    reconstructed_images = pca.inverse_transform(pca_features)
    
    reconstructed_image = reconstructed_images[0].reshape(28, 28)
    
    plt.subplot(1, len(n_components_list) + 1, i + 1)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'n={n_components}')
    plt.axis('off')

    reconstruction_error = np.mean((digit_8_flat[0] - reconstructed_images[0])**2)
    
    print("MSE error for",n_components, "components = ", reconstruction_error)

plt.tight_layout()
plt.show()

# Confusion Matrix 

def metrics_calc(y_true, y_pred):
    num_classes = 10
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Compute confusion matrix
    for x, y in zip(y_true, y_pred):
        confusion_matrix[x, y] += 1
    
    class_metrics = []
    
    for cls in range(num_classes):
        tp = confusion_matrix[cls, cls]
        fp = np.sum(confusion_matrix[:, cls]) - tp
        fn = np.sum(confusion_matrix[cls, :]) - tp
        tn = np.sum(confusion_matrix) - (tp + fp + fn)
        
        # Compute metrics
        accuracy = (tp + tn) / np.sum(confusion_matrix)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'Class': cls,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score })
    
    return confusion_matrix, class_metrics

images = data.reshape(-1, 28, 28)
labels = np.repeat(np.arange(10), 1000)


X_train_img, X_test_img, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

X_train_flat = X_train_img.reshape(X_train_img.shape[0], -1)
X_test_flat = X_test_img.reshape(X_test_img.shape[0], -1)

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)


# data for each model
# 1. MLP 
X_train_mlp = torch.from_numpy(X_train_flat).float()
X_test_mlp = torch.from_numpy(X_test_flat).float()
y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

train_dataset_mlp = torch.utils.data.TensorDataset(X_train_mlp, y_train_tensor)
test_dataset_mlp = torch.utils.data.TensorDataset(X_test_mlp, y_test_tensor)
test_loader_mlp = torch.utils.data.DataLoader(test_dataset_mlp, batch_size=64, shuffle=False)

# 2. CNN 
X_train_cnn = torch.from_numpy(X_train_img).float().unsqueeze(1)
X_test_cnn = torch.from_numpy(X_test_img).float().unsqueeze(1)

train_dataset_cnn = torch.utils.data.TensorDataset(X_train_cnn, y_train_tensor)
test_dataset_cnn = torch.utils.data.TensorDataset(X_test_cnn, y_test_tensor)
test_loader_cnn = torch.utils.data.DataLoader(test_dataset_cnn, batch_size=64, shuffle=False)

# 3. PCA 
X_train_pca_tensor = torch.from_numpy(X_train_pca).float()
X_test_pca_tensor = torch.from_numpy(X_test_pca).float()

train_dataset_pca = torch.utils.data.TensorDataset(X_train_pca_tensor, y_train_tensor)
test_dataset_pca = torch.utils.data.TensorDataset(X_test_pca_tensor, y_test_tensor)
test_loader_pca = torch.utils.data.DataLoader(test_dataset_pca, batch_size=64, shuffle=False)


def goodness(model, test_loader, model_name):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            temp, preds = torch.max(outputs, 1)
            predictions.extend(preds.numpy())
            true_labels.extend(labels.numpy())
    
    cm, metrics = metrics_calc(np.array(true_labels), np.array(predictions))
    
    print(f"\n{model_name} Confusion Matrix:")
    print(cm)
    
    print("\n")
    
    for m in metrics:
        print(f"Class {m['Class']}: "
              f"Precision={m['Precision']:.4f}, "
              f"Recall={m['Recall']:.4f}, "
              f"F1={m['F1 Score']:.4f}")
    
    print(f"\n{model_name} Overall Averages:")
    print(f"Precision: {np.mean([m['Precision'] for m in metrics]):.4f}")
    print(f"Recall: {np.mean([m['Recall'] for m in metrics]):.4f}")
    print(f"F1: {np.mean([m['F1 Score'] for m in metrics]):.4f}")


# calling goodness function for performance check
goodness(Neural_Network, test_loader_mlp, "MLP")
goodness(cnn_model, test_loader_cnn, "CNN")
goodness(mlp_pca_model, test_loader_pca, "MLP+PCA")
goodness(logistic_model, test_loader_pca, "Logistic Regression+PCA")

# ROC curve and AUC
ovr_classifier.classifiers = [c.eval() for c in ovr_classifier.classifiers] 

scores = []
true = []
with torch.no_grad():
    for images, labels in test_loader:
        batch_scores = []
        for i in range(10):
            outputs = ovr_classifier.classifiers[i](images).squeeze()
            batch_scores.append(outputs.numpy())
        batch_scores = np.stack(batch_scores, axis=1)
        scores.append(batch_scores)
        true.append(labels.numpy())
scores = np.concatenate(scores, axis=0)
true = np.concatenate(true, axis=0)


plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'pink'] # used gpt to generate this
for i in range(10):
    fpr, tpr, temp = roc_curve((true == i).astype(int), scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curves for One-vs-Rest Classifier')
plt.legend(loc="lower right")
plt.show()

average_auc = np.mean([auc(fpr, tpr) for i in range(10)])
print(f"Average AUC: {average_auc:.4f}")



"""Question 3: Regression"""

# calling oracle functions
Q3_linear_1_data = q3_linear_1(23684)
Q3_linear_2_data = q3_linear_2(23684)
Q3_stocknet_data = q3_stocknet(23684)

def ols_regression(X, y):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

def ridge_regression(X, y, lambda_=1.0):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    n_features = X.shape[1]
    w = np.linalg.inv(X.T @ X + lambda_ * np.eye(n_features)) @ X.T @ y
    return w

def predict(X, w):
    X = np.hstack([np.ones((X.shape[0], 1)), X]) # Adding bias term (1.0)
    return X @ w
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# making numpy arrays
D1_X_train,D1_y_train,D1_X_test,D1_y_test = Q3_linear_1_data

D1_X_train = np.array(D1_X_train)
D1_y_train = np.array(D1_y_train)
D1_X_test = np.array(D1_X_test)
D1_y_test = np.array(D1_y_test)

D2_X_train,D2_y_train,D2_X_test,D2_y_test = Q3_linear_2_data

D2_X_train = np.array(D2_X_train)
D2_y_train = np.array(D2_y_train)
D2_X_test = np.array(D2_X_test)
D2_y_test = np.array(D2_y_test)

# ols and ridge
w_1_ols = ols_regression(D1_X_train, D1_y_train)
w_1_rr = ridge_regression(D1_X_train, D1_y_train)

w_2_ols = ols_regression(D2_X_train, D2_y_train)
w_2_rr = ridge_regression(D2_X_train, D2_y_train)

# saving csv file
np.savetxt("w_ols_23684.csv", w_2_ols, delimiter = ",")
np.savetxt("w_rr_23684.csv", w_2_rr, delimiter = ",")

# mean squared error

y_pred_1_ols = predict(D1_X_test, w_1_ols)
y_pred_1_rr = predict(D1_X_test, w_1_rr)

y_pred_2_ols = predict(D2_X_test, w_2_ols)
y_pred_2_rr = predict(D2_X_test, w_2_rr)

mse_1_ols = mse(D1_y_test, y_pred_1_ols)
mse_1_rr = mse(D1_y_test, y_pred_1_rr)

mse_2_ols = mse(D2_y_test, y_pred_2_ols)
mse_2_rr = mse(D2_y_test, y_pred_2_rr)

print(y_pred_1_ols)
print(y_pred_1_rr)

print(f"MSE for Linear 1 OLS: {mse_1_ols}")
print(f"MSE for Linear 1 Ridge Regression: {mse_1_rr}")
print(f"MSE for Linear 2 OLS: {mse_2_ols}")
print(f"MSE for Linear 2 Ridge Regression: {mse_2_rr}")

# Stock I got on my SR Number :  BABA

df = pd.read_csv("BABA.csv")
closing_prices = df['Close'].values

mean = np.mean(closing_prices)
std = np.std(closing_prices)

d = (closing_prices - mean) / std # normalise


def linear_svr_dual(X_train, y_train, C=1.0, epsilon=0.5):
    N = X_train.shape[0]
    
    K = np.dot(X_train, X_train.T)  # Linear Kernel
    P = cvxopt.matrix(np.block([[K, -K], [-K, K]]))
    q = cvxopt.matrix(np.hstack([epsilon + y_train, epsilon - y_train]))
    
    G = cvxopt.matrix(np.vstack([np.eye(2*N),-np.eye(2*N)]))
    
    h = cvxopt.matrix(np.hstack([np.ones(2*N) * C, np.zeros(2*N)]))
    
    A = cvxopt.matrix(np.hstack([np.ones(N), -np.ones(N)]), (1, 2*N))
    b = cvxopt.matrix(0.0)

    # using cvxopt for optimization
    soln = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.array(soln['x']).flatten()
    
    return alpha[:N] - alpha[N:]

def rbf_kernel(X, gamma):
    dist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
    return np.exp(-gamma * dist)

def kernelized_svr_dual(X_train, y_train, C=1.0, epsilon=0.1, gamma=0.1):
    N = X_train.shape[0]

    # rbf Kernel
    K = rbf_kernel(X_train, gamma)
    
    P = cvxopt.matrix(np.block([[K, -K], [-K, K]]))
    q = cvxopt.matrix(np.hstack([epsilon + y_train, epsilon - y_train]))
    G = cvxopt.matrix(np.vstack([np.eye(2*N),-np.eye(2*N)]))
    h = cvxopt.matrix(np.hstack([np.ones(2*N) * C, np.zeros(2*N)]))
    
    A = cvxopt.matrix(np.hstack([np.ones(N), -np.ones(N)]), (1, 2*N))
    b = cvxopt.matrix(0.0)

    # usinge cvxopt for optimization
    soln = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.array(soln['x']).flatten()

    return alpha[:N] - alpha[N:]


def linear_svr_prediction(X_train, X_test, y_train, alpha):
    w = np.dot(X_train.T, alpha)
    return np.dot(X_test, w)

def kernel_svr_prediction(X_train, X_test, y_train, alpha, gamma):
    K = np.exp(-gamma * np.sum((X_train[:, None] - X_test) ** 2, axis=2))
    return np.dot(K.T, alpha)


N = len(d)
time = [7,30,90]
gammas = [1.0,0.1,0.01,0.001]

for t in time:
    temp = [d[i:i+t] for i in range(N - t)]
    X = np.array(temp)
    y = d[t:]

    mid = len(X) // 2
    X_train, X_test = X[:mid], X[mid:]
    y_train, y_test = y[:mid], y[mid:]
    
    # calling linear svr
    alpha_linear = linear_svr_dual(X_train, y_train)
    y_pred_linear = linear_svr_prediction(X_train, X_test, y_train, alpha_linear)
    rmse_linear = np.sqrt(mse(y_test, y_pred_linear))

    # plotting linear svr
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="true closing price", color='blue')
    plt.plot(y_pred_linear, label="predicted closing price", color='green')
    plt.xlabel("days")
    plt.ylabel("closing price")
    plt.title(f"Plot for Linear SVR and time = {t}  ")
    plt.legend()
    plt.grid(True)
    plt.show()

    for gamma in gammas:
        alpha_rbf = kernelized_svr_dual(X_train, y_train, gamma)
        y_pred_rbf = kernel_svr_prediction(X_train, X_test, y_train, alpha_rbf, gamma)
        rmse_rbf = np.sqrt(mse(y_test, y_pred_rbf))

        y_avg = np.array([np.mean(y_test[max(0, i - t):i]) for i in range(len(y_test))])

        # plotting kernel svr
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label="true closing price", color='blue')
        plt.plot(y_pred_rbf, label="predicted closing price", color='green')
        plt.plot(y_avg, label="average price(last t days)", color='red', linestyle='dashed')
        
        plt.xlabel("days")
        plt.ylabel("closing price")
        plt.title(f"graph for (t={t}, γ={gamma})")
        plt.legend()
        plt.grid(True)
        plt.show()