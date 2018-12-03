from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import StandardScaler
import warnings

def report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TG = cm[1][1]
    print('True positive = ', TP)
    print('False positive = ', FP)
    print('False negative = ', FN)
    print('True negative = ', TG)
    # precision = TP / (TP + FP)
    print('precision = {:.2f}%'.format(TP / (TP + FP) * 100))
    # recall = TP / (TP + FN)
    print('recall = {:.2f}%'.format(TP / (TP + FN) * 100))


def fit_svm(X, y, comment, use_x_normalization, kernel=None):
    print("------------------------------")
    print(comment)
    print("------------------------------")
    np.random.seed(1)
    if use_x_normalization:
        X = StandardScaler().fit_transform(X)

    train_scores = np.array([])
    val_scores = np.array([])
    
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if kernel is None:
            clf = svm.SVC()
        else:
            clf = svm.SVC(kernel=kernel)
        clf.fit(X_train, y_train)

        print('start to calculate p value')
        score, permutation_scores, pvalue = permutation_test_score(clf, X_train, y_train, scoring="accuracy", cv=kf, n_permutations=10, n_jobs=1)
        print(score, permutation_scores, pvalue)
        
        train_scores = np.append(train_scores, clf.score(X_train, y_train) * 100)
        val_scores = np.append(val_scores, clf.score(X_val, y_val) * 100)

    print('Training accuracy: {:.2f}%'.format(np.mean(train_scores)))
    print('Validation accuracy: {:.2f}%'.format(np.mean(val_scores)))
    print()


def main():
    X_no_uni = pd.read_csv('./dataset/gradcafe/cs_preprocessed_X.csv', usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).values
    X_with_uni = pd.read_csv('./dataset/gradcafe/pnp_x.csv', header=None).values
    y = pd.read_csv('./dataset/gradcafe/cs_preprocessed_Y.csv').values.reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(X_no_uni, y, test_size=0.1, random_state=0)

    # fit_svm(X_train, y_train, "Without normalization on X no uni", False)
    # fit_svm(X_train, y_train, "With normalization on X no uni", True)

    X_train, X_test, y_train, y_test = train_test_split(X_with_uni, y, test_size=0.1, random_state=0)

    # fit_svm(X_train, y_train, "Without normalization on X with uni", False)
    fit_svm(X_train, y_train, "With normalization on X with uni", True)

    # try different kernels
    fit_svm(X_train, y_train, "With normalization on X with uni with linear kernel", True, 'linear')
    fit_svm(X_train, y_train, "With normalization on X with uni with poly kernel", True, 'poly')
    fit_svm(X_train, y_train, "With normalization on X with uni with rbf kernel (default)", True, 'rbf')
    fit_svm(X_train, y_train, "With normalization on X with uni with sigmoid kernel", True, 'sigmoid')
    fit_svm(X_train, y_train, "With normalization on X with uni with precomputed kernel", True, 'precomputed')

    # Normalization with uni is better
    X_train, X_test, y_train, y_test = train_test_split(X_with_uni, y, test_size=0.1, random_state=0)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    np.random.seed(1)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    
    training_accuracy = clf.score(X_train, y_train) * 100
    testing_accuracy = clf.score(X_test, y_test) * 100

    print("------------------------------")
    print("Results with normalization on testing set")
    print("------------------------------")
    print('Training accuracy: {:.2f}%'.format(training_accuracy))
    print('Testing accuracy: {:.2f}%'.format(testing_accuracy))
    report(y_test, clf.predict(X_test))
    print()
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

