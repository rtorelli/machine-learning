import sys
sys.path.append('libsvm/python/')
from svmutil import svm_train, svm_predict
'''
Homework5: support vector machine classifier

You need to use two functions 'svm_train' and 'svm_predict'
from libsvm library to start your homework. Please read the 
readme.txt file carefully to understand how to use these 
two functions.

'''

def svm_with_diff_c(train_label, train_data, test_label, test_data):
    cost = [0.01, 0.1, 1, 2, 3, 5]
    for c in cost:
        libsvm_options = '-c ' + str(c)
        print('c =', c, ' ', end="")
        model = svm_train(train_label, train_data, libsvm_options)
        print('sv =', model.get_nr_sv(), ' ', end='')
        predicted_label, test_acc, decision_values = svm_predict(test_label, test_data, model)

def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
    kernel = [0, 1, 2]
    for k in kernel:
        libsvm_options = '-t ' + str(k)
        print('k =', k, ' ', end="")
        model = svm_train(train_label, train_data, libsvm_options)
        print('sv =', model.get_nr_sv(), ' ', end='')
        predicted_label, test_acc, decision_values = svm_predict(test_label, test_data, model)
