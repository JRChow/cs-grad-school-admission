import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pygam import LogisticGAM
import warnings

def fit_gam(X, y, comment, use_x_normalization):
    print("------------------------------")
    print(comment)
    print("------------------------------")
    np.random.seed(0)
    if use_x_normalization:
        X = StandardScaler().fit_transform(X)

    train_scores = np.array([])
    val_scores = np.array([])
    
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        clf = LogisticGAM()
        clf.fit(X_train, y_train)
        
        train_scores = np.append(train_scores, clf.accuracy(X_train, y_train) * 100)
        val_scores = np.append(val_scores, clf.accuracy(X_val, y_val) * 100)

    print('Training accuracy: {:.2f}%'.format(np.mean(train_scores)))
    print('Validation accuracy: {:.2f}%'.format(np.mean(val_scores)))
    print()


def main():
    X = pd.read_csv('./dataset/gradcafe/cs_preprocessed_X.csv', usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).values
    # X = pd.read_csv('./dataset/gradcafe/pnp_x.csv', header=None).values
    y = pd.read_csv('./dataset/gradcafe/cs_preprocessed_Y.csv').values.reshape(-1)

    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    fit_gam(X_train, y_train, "Without normalization on X", False)
    fit_gam(X_train, y_train, "With normalization on X", True)

    # Normalization is better
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    np.random.seed(0)
    clf = LogisticGAM()
    clf.fit(X_train, y_train)
    
    training_accuracy = clf.accuracy(X_train, y_train) * 100
    testing_accuracy = clf.accuracy(X_test, y_test) * 100

    print("------------------------------")
    print("Results with normalization on testing set")
    print("------------------------------")
    print('Training accuracy: {:.2f}%'.format(training_accuracy))
    print('Testing accuracy: {:.2f}%'.format(testing_accuracy))
    print()
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

