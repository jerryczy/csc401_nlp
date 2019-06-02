from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from scipy import stats
import numpy as np
import argparse
import sys
import os
import csv

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    result = 0
    total_sum = np.sum(C) 
    if total_sum > 0:
        result = np.sum(np.diag(C))/total_sum
    return result

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    result = np.zeros(4)
    for i in range(3):
        k_sum = np.sum(C[i])
        if k_sum > 0:
            result[i] = C[i][i]/k_sum
    return result

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    result = np.zeros(4)
    for i in range(3):
        k_sum = 0
        for j in range(3):
            k_sum += np.sum(C[i][j])
        if k_sum > 0:
            result[i] = C[i][i]/k_sum
    return result
    
clf_list = [LinearSVC(), SVC(gamma=2), RandomForestClassifier(max_depth=5), \
    MLPClassifier(alpha=0.05), AdaBoostClassifier()]

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')
    with np.load(filename) as j:
        data = j['arr_0']
    x = np.zeros((len(data), 173))
    y = np.zeros((len(data), 1), dtype=int)
    for i in range(len(data)):
        x[i], y[i] = np.split(data[i], [173])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    all_acc = []
    csvfile = open('a1_3.1.csv', 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(len(clf_list)):
        clf_list[i].fit(X_train, y_train.ravel())
        prediction = clf_list[i].predict(X_test)
        # print('done training', i)
        c = confusion_matrix(prediction, y_test)
        acc = accuracy(c)
        all_acc.append(acc)
        rec = recall(c)
        prc = precision(c)
        writer.writerow([i + 1, acc, rec[0], rec[1], rec[2], rec[3], \
            prc[0], prc[1], prc[2], prc[3], c[0][0], c[0][1], c[0][2], c[0][3], \
            c[1][0], c[1][1], c[1][2], c[1][3], c[2][0], c[2][1], c[2][2], c[2][3], \
            c[3][0], c[3][1], c[3][2], c[3][3]])
    csvfile.close()

    iBest = all_acc.index(max(all_acc)) + 1

    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''
    print('TODO Section 3.2')
    clf = clf_list[iBest-1]
    accuracies = []
    for i in [1, 5, 10, 15, 20]:
        x, y = random_train_set([X_train, y_train], i*1000)
        if i == 1:
            X_1k, y_1k = x, y
        clf.fit(x, y.ravel())
        prediction = clf[i].predict(X_test)
        # print('done training', i)
        c = confusion_matrix(prediction, y_test)
        acc = accuracy(c)
        accuracies.append(acc)

    with open('a1_3.2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(accuracies)

    return (X_1k, y_1k)

def random_train_set(data_list, num):
    rng_state = np.random.get_state()
    result = []
    for data in data_list:
        np.random.set_state(rng_state)
        np.random.shuffle(data)
        result.append(data[:num])
    return result
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')
    # p values
    csvfile = open('a1_3.3.csv', 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    k5 = np.zeros((32000, 5))
    k5_1 = np.zeros((1000, 5))
    for num in [5, 10, 20, 30, 40, 50]:
        pp_val = [num]
        for test_set in [[X_1k, y_1k], [X_train, y_train]]:
            selector = SelectKBest(f_classif, k=num)
            X_new = selector.fit_transform(test_set[0], test_set[1])
            select = selector.get_support()
            pp = selector.pvalues_
            for item in range(len(pp)):
                if select[item]:
                    pp_val.append((item, pp[item]))
            print('done select and calculation', num, len(test_set[1]))
            if num == 5:
                X_test_feature = selector.transform(X_test)
                if len(X_new) == 1000:
                    k5_1 = X_new
                else:
                    k5 = X_new
        writer.writerow(pp_val)

    #select best 5 features
    clf = clf_list[i-1]
    acc = []
    for data in [[k5_1, y_1k], [k5, y_train]]:
        clf.fit(data[0], data[1].ravel())
        prediction = clf.predict(X_test_feature)
        c = confusion_matrix(prediction, y_test)
        acc.append(accuracy(c))
    writer.writerow(acc)
    csvfile.close()

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    with np.load(filename) as j:
        data = j['arr_0']
    x = np.zeros((len(data), 173))
    y = np.zeros((len(data), 1), dtype=int)
    for index in range(len(data)):
        x[index], y[index] = np.split(data[index], [173])

    csvfile = open('a1_3.4.csv', 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    kf = KFold(n_splits=5, shuffle=True)
    all_acc = [] # every element in this is a list containing accuracies for a fold
    for train_index, test_index in kf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc = []
        for clf in clf_list:
            clf.fit(X_train, y_train.ravel())
            print('done training')
            prediction = clf.predict(X_test)
            c = confusion_matrix(prediction, y_test)
            acc.append(accuracy(c))
        writer.writerow(acc)
        all_acc.append(acc)
    print('done all trainings')
    best_acc = []
    for index in range(5):
        best_acc.append(all_acc[index][i-1])
    p_val = []
    for c in range(5):
        clf_acc = []
        if c != i-1:
            for index in range(5):
                clf_acc.append(all_acc[index][c])
            p_val.append(stats.ttest_rel(best_acc, clf_acc)[1])
    writer.writerow(p_val)
    csvfile.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # complete each classification experiment, in sequence.
    (X_train, X_test, y_train, y_test,iBest) = class31(args.input)
    (X_1k, y_1k) = class32(X_train, X_test, y_train, y_test, iBest)
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.input, iBest)
    print('Done!')
