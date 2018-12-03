import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import warnings

def main():
    X = pd.read_csv('../dataset/gradcafe/cs_preprocessed_X.csv', usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).values
    university_names = pd.read_csv('../dataset/gradcafe/cs_preprocessed_X.csv', usecols=[5]).values
    year = pd.read_csv('../dataset/gradcafe/cs_preprocessed_X.csv', usecols=[4]).values
    y = pd.read_csv('../dataset/gradcafe/cs_preprocessed_y.csv').values.reshape(-1)

    le_un = LabelEncoder()
    le_un.fit(university_names)
    le_encoded_un = le_un.transform(university_names).reshape(-1, 1)
    ohe_un = OneHotEncoder(sparse=False)
    ohe_un.fit(le_encoded_un)
    ohe_encoded_un = ohe_un.transform(le_encoded_un)
    X = np.hstack((X, ohe_encoded_un))

    ohe_yr = OneHotEncoder(sparse=False)
    ohe_yr.fit(year)
    ohe_encoded_yr = ohe_yr.transform(year)

    X = np.hstack((X, ohe_encoded_yr))

    print(X.shape)

    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    training_accuracies = []
    testing_accuracies = []

    kf = KFold(n_splits=10, shuffle=True)
    for train_index, val_index in kf.split(X):
        X_train, X_test = X[train_index], X[val_index]
        y_train, y_test = y[train_index], y[val_index]

        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        training_accuracy = clf.score(X_train, y_train) * 100
        testing_accuracy = clf.score(X_test, y_test) * 100

        training_accuracies.append(training_accuracy)
        testing_accuracies.append(testing_accuracy)

    print("------------------------------")
    print("Results with one hot on testing set")
    print("------------------------------")
    print('Training accuracy: {:.2f}%'.format(np.mean(training_accuracies)))
    print('Testing accuracy: {:.2f}%'.format(np.mean(testing_accuracies)))
    print()
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
