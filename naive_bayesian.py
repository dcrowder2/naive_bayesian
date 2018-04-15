from sklearn import model_selection
from sklearn import metrics
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
    return model_selection.train_test_split(array[:stop], train_size=".2")


# returns an array for a single instance with the probabilities label | x for each label
def bayes_rule(array_x, label):
    return []


# returns a 2d array, size (2, feature_size), which is for conditional probability of x | label
def conditional_probability(array):
    probability_array = [np.zeros(len(array[0][1:])), np.zeros(len(array[0][1:]))]
    spam = []
    ham = []
    for i in range(len(array)):
        if array[0] == -1:
            spam.append(array[i])
        else:
            ham.append(array[i])
    spam_divisor = np.full((1, len(spam[0][1:])), len(spam))
    ham_divisor = np.full((1, len(ham[0][1:])), len(ham))
    spam_prob = np.count_nonzero(spam, axis=0) / spam_divisor
    ham_prob = np.count_nonzero(ham, axis=0) / ham_divisor


def run():
    print("Loading file")
    features = init("SpamInstances.txt")


if __name__ == "__main__":
    print("Starting")
    run()
