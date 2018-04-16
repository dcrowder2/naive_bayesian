# naive_bayesian.py
# Author: Dakota Crowder
# Email: dcrowder2@alaska.edu
# A naive bayesian classifier that specifically uses data from SpamHero in a text file called "SpamInstance.txt"
# to run tests on instances from 100 to 1900 and then the whole file of 15498 instances
# written for CSCE A415 Machine Learning
# University of Alaska Anchorage
# Professor Martin Cenek

from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np


# Initializes two array [label, features..] using the file name passed, special for the format used in SpamInstances.txt
# which are sent into even_spread, so that the array is evenly distributed spam and not spam
def init(file_name):
    with open(file_name) as file:
        print("File open, reading")
        line = file.readline()  # Get the first line which is just the number of emails and number of features
        spam = []
        ham = []
        for line in file:
            if line != "\n":
                features = []
                line = line.split(" ")
                features.append(int(line[1]))
                for i in range(len(line[2])-1):  # -1 because there is a \n character at the end of the read line
                    features.append(int(line[2][i]))
                if features[0] == 1:
                    spam.append(features)
                else:
                    ham.append(features)
        print("Finished reading")
        file.close()
    return even_spread(spam, ham)


# takes in two arrays, of varying size, and makes one array with the elements of the two arrays split evenly in the new
# array until one array is empty then it just places the remaining elements in the return array
def even_spread(array_a, array_b):
    ret_array = []
    i = 0
    k = 0
    while i < (len(array_a) + len(array_b)):
        if k < len(array_a):
            ret_array.append(array_a[k])
            i += 1
        if k < len(array_b):
            ret_array.append(array_b[k])
            i += 1
        k += 1
    return ret_array


# Returns 2 arrays for the train test split using an index for how many instances to use with 80/20 split for train test
def train_test_split(array, stop):
    return model_selection.train_test_split(array[:stop], train_size=.2, test_size=.8)


# returns an array for the label with the highest probability and the probability for all instances
def bayes_rule(test, data):
    spam_cond_prob, ham_cond_prob, spam_prob, ham_prob = conditional_probability(data)
    hypothesis = []
    for instance in test:
        prob_spam = 0
        prob_ham = 0
        for feature in range(len(instance[1:]) - 1):
            if instance[feature + 1] == 1:
                prob_spam += np.log(spam_cond_prob[feature + 1])
                prob_ham += np.log(ham_cond_prob[feature + 1])
            else:
                prob_spam += np.log((1 - spam_cond_prob[feature + 1]))
                prob_ham += np.log((1 - ham_cond_prob[feature + 1]))
        bayes = [prob_ham + np.log(ham_prob), prob_spam + np.log(spam_prob)]
        hypothesis.append((instance[0], np.argmax(bayes), bayes[int(np.argmax(bayes))]))
    return hypothesis


# returns 2 arrays for the probability of xi == 1 | label for spam and ham, as well as prob of spam and ham
def conditional_probability(array):
    spam = []
    ham = []
    for item in array:
        if item[0] == -1:
            ham.append(item[1:])
        else:
            spam.append(item[1:])
    spam_prob = ((np.count_nonzero(spam, axis=0) + 1) / ((len(spam) * 1.) + 1))
    ham_prob = ((np.count_nonzero(ham, axis=0) + 1) / ((len(ham) * 1.) + 1))
    return spam_prob, ham_prob, (len(spam) / (len(array) * 1.)), (len(ham) / (len(array) * 1.))


# takes in an array with structure like this, [[label, guess, prob], ...] , and returns the false positive rate and true
# positive rate to be used for a ROC curve, and the accuracy
def roc_calc(array):
    false_pos = 0
    true_pos = 0
    false_neg = 0
    true_neg = 0
    for instance in array:
        if instance[1] == 1 and instance[0] == 1:
            true_pos += 1
        elif instance[1] == 0 and instance[0] == -1:
            true_neg += 1
        elif instance[1] == 1 and instance[0] == -1:
            false_pos += 1
        else:
            false_neg += 1
    return (false_pos / (false_pos + true_neg * 1.)), (true_pos / (true_pos + false_neg * 1.)), \
           (true_neg + true_pos) / (len(array) * 1.)


def run():
    print("Loading file")
    features = init("SpamInstances.txt")
    FPR = []
    TPR = []
    accuracy_array = []
    instances_count = []
    for i in range(19):
        stop = 100 * (i+1)
        instances_count.append(stop)
        print("Using " + str(stop) + " instances")
        print("Splitting test and train lists")
        test, data = train_test_split(features, stop)
        print("Getting probabilities")
        temp = np.array(bayes_rule(test, data))
        print("Calculating FPR, TPR, and Accuracy")
        fpr, tpr, accuracy = roc_calc(temp)
        FPR.append(fpr)
        TPR.append(tpr)
        accuracy_array.append(accuracy)
        print("Accuracy = ", end="")
        print(accuracy)
    # final run
    stop = len(features)
    print("Using " + str(stop) + " instances")
    print("Splitting test and train lists")
    test, data = train_test_split(features, stop)
    print("Getting probabilities")
    temp = np.array(bayes_rule(test, data))
    print("Calculating FPR, TPR, and Accuracy")
    fpr, tpr, accuracy = roc_calc(temp)
    FPR.append(fpr)
    TPR.append(tpr)
    print("Accuracy = ", end="")
    print(accuracy)
    plt.figure()
    plt.plot(FPR, TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.figure()
    plt.plot(instances_count, accuracy_array)
    plt.xlabel('Instance count')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for each instance count')
    plt.show()


if __name__ == "__main__":
    print("Starting")
    run()
