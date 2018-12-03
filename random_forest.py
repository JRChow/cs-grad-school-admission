import pandas as pd
import numpy as np
import warnings
import json
import pydot
from multiprocessing import Pool
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from sys import platform as sys_pf
from sklearn.metrics import confusion_matrix

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

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

#draw a tree of a random forest by id
def draw_tree(rf, id, feature_list):
    # Pull out one tree from the forest
    tree = rf.estimators_[id]
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')

def sub_module(input):
    max_depth,min_samples_split,min_samples_leaf,max_features,train_features,train_labels= input
    accuracies = np.array([])
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, val_index in kf.split(train_features):
        X_train, X_val = train_features[train_index], train_features[val_index]
        y_train, y_val = train_labels[train_index], train_labels[val_index]
        rf = RandomForestClassifier(n_estimators = 100,
                                max_depth = max_depth,
                                min_samples_split = min_samples_split,
                                min_samples_leaf = min_samples_leaf,
                                max_features = max_features,
                               criterion = "entropy",
                               random_state = 41
                               )
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_val)
        predictions = predictions.reshape((predictions.shape[0],1))
        mean_average_precision = np.average(np.equal(predictions,y_val).astype(np.float))
        accuracies = np.append(accuracies,mean_average_precision)
    mean_average_precision = np.mean(accuracies)
    return(mean_average_precision)

def rf(data_x,data_y,is_searching_parameters = False):
    output = {}
    labels = np.array(data_y)
    feature_list = data_x.columns
    features = np.array(data_x)
    #split data into traning and testing
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 233)

    best_accuracy = 0.0
    best_parameters = ()

    if is_searching_parameters:
        exp_num = 0
        p = Pool(10)
        parameters = []
        for max_depth in [15]:
            # print("exploring max depth",max_depth)
            for min_samples_split in [2]:
                for min_samples_leaf in [1,3,4,5]:
                    for max_features in [11,13,15]:
                        parameters.append((max_depth,min_samples_split,min_samples_leaf,max_features,train_features,train_labels))

                        # if mean_average_precision > best_accuracy:
                        #     best_accuracy = mean_average_precision
                        #     best_parameters = [max_depth,min_samples_split,min_samples_leaf,max_features]

                        # exp_num += 1
                        # output[mean_average_precision] = [max_depth,min_samples_split,min_samples_leaf,max_features]

        results = p.map(sub_module,tqdm(parameters))
        print('experiment number, max_depth, min_samples_leaf, max_features, cross validaiton accuracy')
        for i,(max_depth,min_samples_split,min_samples_leaf,max_features,_,_) in enumerate(parameters):
            print('%d,%d,%d,%d,%f'%(i,max_depth,min_samples_leaf,max_features,results[i]))

        # plt.plot(range(1,16),results)
        # plt.legend("validation_accuracy ~ max_features " ,loc='upper left')
        # plt.title("validation_accuracy ~ max_features")
        # plt.savefig("random_forest_exp/max_features_vs_validation_accuracy.png")
        json.dump(output,open("random_forest_parameters.json",'w+'))


    #build a random forest regressor with 2000 trees
    else:
        rf = RandomForestClassifier(
                                   n_estimators = 8000,
                                     criterion = "entropy",
                                   max_depth = 17,
                                   min_samples_leaf=6,
                                   max_features = 15,
                                   random_state = 42
                                   )
        rf.fit(train_features, train_labels)

        # use the forest's predict method on the test data
        predictions = rf.predict(test_features)
        predictions = predictions.reshape((predictions.shape[0],1))
        #calculate mean average precision
        mean_average_precision = np.average(np.equal(predictions,test_labels).astype(np.float))
        print('Mean Average Precision', mean_average_precision)
        # draw_tree(rf,1000,feature_list)

        #use the forest's predict method on the test data
        predictions = rf.predict(train_features)
        predictions = predictions.reshape((predictions.shape[0],1))
        #calculate mean average precision
        mean_average_precision = np.average(np.equal(predictions,train_labels).astype(np.float))
        print('Mean Average Precision', mean_average_precision)
        draw_tree(rf,0,feature_list)
        importance = (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_list),
                 reverse=True))
        print('relative importance, parameter name')
        for (score, para) in importance:
            print('%f,%s'%(score,para))
        report(train_labels,predictions)


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    X = pd.read_csv('dataset/gradcafe/cs_preprocessed_X.csv', usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    feature_list = X.columns
    university_names = pd.read_csv('dataset/gradcafe/cs_preprocessed_X.csv', usecols=[5]).values
    year = pd.read_csv('dataset/gradcafe/cs_preprocessed_X.csv', usecols=[4]).values
    y = pd.read_csv('dataset/gradcafe/cs_preprocessed_Y.csv')

    # le = LabelEncoder()
    # le.fit(university_names)
    # le_encoded = le.transform(university_names).reshape(-1, 1)
    # ohe = OneHotEncoder(sparse=False)
    # ohe.fit(le_encoded)
    # ohe_encoded = ohe.transform(le_encoded)
    # X = np.hstack((X, ohe_encoded))
    #
    # ohe_yr = OneHotEncoder(sparse=False)
    # ohe_yr.fit(year)
    # ohe_encoded_yr = ohe_yr.transform(year)
    # X = np.hstack((X,ohe_encoded_yr))
    #
    # pca = PCA(n_components=20)
    # principalComponents = pca.fit_transform(X)
    # print(feature_list)
    # for i in range(len(pca.components_[0])):
    #     if i < len(feature_list):
    #         print(feature_list[i],',',end='')
    #     else:
    #         print(i,',',end='')
    # print()
    # X = StandardScaler(X)
    # for index,i in enumerate(pca.components_):
    #     for j in i:
    #         print("%.1f"%(j),',',end='')
    #     print()
    # exit()
    # principalDf = pd.DataFrame(data = principalComponents)
    # print(principalDf)
    rf(X,y)

if __name__ == '__main__':
    main()
