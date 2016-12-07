## Import packages and dependencies
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import random
random.seed(14)

# Load data and clean it up
digits = datasets.load_digits()
target_names = digits.target_names
X_digits = digits.data
y_digits = digits.target
# Change y_digits so that digits 0-4 are 0 
# and those 5-9 are 1
y_digits_round = [0] * len(y_digits)
for i in range(len(y_digits)):
    if y_digits[i]>4:
        y_digits_round[i] = 1
# Record sample size and image dims        
n_samples, h, w = digits.images.shape          
# Split the data into testing and training data 
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits_round, test_size=0.25, random_state=42)

# Pick number of components to keep from PCA (capture 85% of variance)
n_components = 16

# Build PCA
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

pca_digits = pca.components_.reshape((n_components, h, w))

# Transform using PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a LR classification model
grid = {
        'C': np.power(10.0, np.arange(-10, 10))
         , 'solver': ['newton-cg']
    }
clf = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
gs = GridSearchCV(clf, grid)
gn = GridSearchCV(clf, grid)
# Fit LR
gs.fit(X_train_pca, y_train)
gn.fit(X_train, y_train)

# Predict
ypca_pred = gs.predict(X_test_pca)
y_pred = gn.predict(X_test)

# Summarize results using confusion matrix
C = confusion_matrix(y_test, ypca_pred, labels=range(2))
Cn = confusion_matrix(y_test, y_pred, labels=range(2))
print(C)
print(Cn)
print(np.diag(C) / map(float, np.sum(C,1)))
print(np.diag(Cn) / map(float, np.sum(Cn,1)))

# Demonstrate the images before and after PCA transform
# Function below was not written by me
def plot_gallery(images, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())

plot_gallery(X_test, h, w)
plot_gallery(pca_digits, h, w)

plt.show()

# Get FPR and TPR for ROC curve and AUC
fpr, tpr, _ = metrics.roc_curve(np.array(y_test), gs.predict_proba(X_test_pca)[:,1])
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic after PCA')
plt.legend(loc="lower right")
plt.show()

