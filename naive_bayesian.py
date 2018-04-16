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
    return model_selection.train_test_split(array[:stop], train_size=.2, test_size=.8)


# returns an array for the label with the highest probability and the probability for all instances
def bayes_rule(test, data):
    spam_cond_prob, ham_cond_prob, spam_prob, ham_prob = conditional_probability(data)
    hypothesis = []
    for instance in test:
        prob_spam = 1
        prob_ham = 1
        for feature in range(len(instance) - 1):
            if instance[feature + 1] == 1:
                prob_spam *= spam_cond_prob[feature + 1]
                prob_ham *= ham_cond_prob[feature + 1]
            else:
                prob_spam *= 1 - spam_cond_prob[feature + 1]
                prob_ham *= 1 - ham_cond_prob[feature + 1]
        bayes = [prob_ham * ham_prob, prob_spam * spam_prob]
        hypothesis.append((instance[0], np.argmax(bayes), bayes[int(np.argmax(bayes))]))
    return hypothesis


# returns 2 arrays for the probability of xi == 1 | label for spam and ham, as well as prob of spam and ham
def conditional_probability(array):
    spam = []
    ham = []
    for item in array:
        if item[0] == -1:
            ham.append(item)
        else:
            spam.append(item)
    spam_prob = np.count_nonzero(spam, axis=0) + 1 / (len(spam) * 1.) + 1
    ham_prob = np.count_nonzero(ham, axis=0) + 1 / (len(ham) * 1.) + 1
    return spam_prob, ham_prob, (len(spam) / (len(array) * 1.)), (len(ham) / (len(array) * 1.))


def run():
    print("Loading file")
    features = init("SpamInstances.txt")
    print("Splitting test and train lists")
    test, data = train_test_split(features, 100)
    print("Getting probabilities")
    temp = np.array(bayes_rule(test, data))
    print(temp)
    print(temp.shape)


if __name__ == "__main__":
    print("Starting")
    run()
