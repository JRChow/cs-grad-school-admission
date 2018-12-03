import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import threading
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings

results = []
results_without_norm = []

class Fit_knn(threading.Thread):
    def __init__(self, X, y, comment, use_x_normalization, neighbors):
        super(Fit_knn, self).__init__()
        self.X = X
        self.y = y
        self.comment = comment
        self.use_x_normalization = use_x_normalization
        self.neighbors = neighbors

    def run(self):
        X = self.X
        y = self.y
        comment = self.comment
        use_x_normalization = self.use_x_normalization
        neighbors = self.neighbors
        np.random.seed(0)
        if use_x_normalization:
            X = StandardScaler().fit_transform(X)

        train_scores = np.array([])
        val_scores = np.array([])

        kf = KFold(n_splits=5, shuffle=True)
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            clf = KNeighborsClassifier(n_neighbors=neighbors)
            clf.fit(X_train, y_train)

            train_scores = np.append(train_scores, clf.score(X_train, y_train) * 100)
            val_scores = np.append(val_scores, clf.score(X_val, y_val) * 100)

        print("------------------------------")
        print(comment)
        print("------------------------------")

        training_accuracy = np.mean(train_scores)
        validation_accuracy = np.mean(val_scores)
        print('Training accuracy: {:.2f}%'.format(training_accuracy))
        print('Validation accuracy: {:.2f}%'.format(validation_accuracy))
        if use_x_normalization:
            results.append((neighbors, training_accuracy, validation_accuracy))
        else:
            results_without_norm.append((neighbors, training_accuracy, validation_accuracy))
        print()


def main():
    # X = pd.read_csv('./dataset/gradcafe/cs_preprocessed_X.csv', usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).values
    X = pd.read_csv('./dataset/gradcafe/pnp_2_0_x.csv', header=None).values
    y = pd.read_csv('./dataset/gradcafe/cs_preprocessed_y.csv').values.reshape(-1)

    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    threads = []
    for i in range(1, 50): 
        thread = Fit_knn(X_train, y_train, "Without normalization on X with neighbors = %d" % i, False, i)
        threads.append(thread)
        thread.start()
        thread_norm = Fit_knn(X_train, y_train, "With normalization on X with neighbors = %d" % i, True, i)
        thread_norm.start()
        threads.append(thread_norm)

    for thread in threads:
        thread.join()

    # print('========results========')
    # print(results)
    # print('========results========')

    plt.xlabel('N (number of nearest neighbors)')
    plt.ylabel('Accuracy')
    neighbors = []
    train_scores = []
    val_scores = []
    results_sorted = sorted(results, key = lambda x: x[0])

    for result in results_sorted:
        neighbors.append(result[0])
        train_scores.append(result[1])
        val_scores.append(result[2])

    plt.plot(neighbors, train_scores)
    plt.plot(neighbors, val_scores)

    neighbors = []
    train_scores = []
    val_scores = []
    results_sorted = sorted(results_without_norm, key = lambda x: x[0])

    for result in results_sorted:
        neighbors.append(result[0])
        train_scores.append(result[1])
        val_scores.append(result[2])

    plt.plot(neighbors, train_scores)
    plt.plot(neighbors, val_scores)

    plt.legend(['training accuracy (normalized)', 'validation accuracy (normalized)', 'training accuracy', 'validation accuracy'], loc='upper left')
    plt.show()

    # find the best model with normalization
    best = max(results, key=lambda x: x[2])
    print('best model with normalization:', best)

    # find the best model without normalization
    best = max(results_without_norm, key=lambda x: x[2])
    print('best model without normalization:', best)

    # Normalization is better
    # X_train = StandardScaler().fit_transform(X_train)
    # X_test = StandardScaler().fit_transform(X_test)

    # np.random.seed(0)
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf.fit(X_train, y_train)
    # 
    # training_accuracy = clf.score(X_train, y_train) * 100
    # testing_accuracy = clf.score(X_test, y_test) * 100

    # print("------------------------------")
    # print("Results with normalization on testing set")
    # print("------------------------------")
    # print('Training accuracy: {:.2f}%'.format(training_accuracy))
    # print('Testing accuracy: {:.2f}%'.format(testing_accuracy))
    # print()
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

