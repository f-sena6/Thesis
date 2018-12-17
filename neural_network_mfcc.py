#########################################
#
#  Fred Sena
#  Thesis
#  Predict Popularity: Neural Network
#
#########################################


from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score

def get_pca(x):

    pca = PCA(n_components=0.99)   # Reduce by 99%
    x_pca = pca.fit_transform(x)

    print "PCA: ", x_pca.shape
    return x_pca

def count_ones(songs):
    count = 0
    for i in songs:
        if i[-1] == 1:
            count += 1

    print count


def benchmark(songs, count):

    mean_list = []

    for i in range(count):
        mean = neural_network_classify(songs)
        mean_list.append(mean)

    score = '%.2f%%' % (np.mean(mean_list)*100)
    print "\nAverage for {} runs: {}".format(count, score), '(+/- %.2f%%)' % (np.std(mean_list)*100)


def neural_network_classify(data):

    x = data[:, 0:-1]
    y = data[:, -1]

    x, y = shuffle(x, y)

    #x = x/np.linalg.norm(x) # Normalize the data
    #x = preprocessing.scale(x)

    ###########
    # Get PCA #
    ###########
    #x = get_pca(x)  # Reduce dataset here

    ###################
    # Set up the data #
    ###################
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)

    ######################
    # Set up the Network #
    ######################
    clf = MLPClassifier(max_iter=1000, solver='adam', activation='relu', hidden_layer_sizes=(128, 64, 32,)) # lbfgs;   logistic
    clf.out_activation_ = 'relu'
    print "Output Activation: ", clf.out_activation_
    ####################
    # Cross Validation #
    ####################
    print "Computing Cross Validation: "
    cross_validation_score = cross_val_score(clf, x, y, cv=10, n_jobs=-1)
    print "Fitting the model..."

    #################
    # Fit the model #
    #################
    clf.fit(x_train, y_train)

    ###########
    # Predict #
    ###########
    test_score = clf.score(x_test, y_test)
    prediction = clf.predict(x_test)


    print "Y :", y_test
    print "Y^:", prediction
    print 
    print "Score: ", test_score
    print "F1 Score: ", f1_score(y_test, prediction, average='weighted') 


    print "KFold Scores: ", cross_validation_score
    print
    print "KFold Mean: %.2f%%" % (cross_validation_score.mean() * 100), '(+/- %.2f%%)' % (np.std(cross_validation_score)*100)

    return cross_validation_score.mean()


file_list = ['MFCC_Files/full_pca_metallica.txt', 'MFCC_Files/three_days_grace_pca.txt']

data = np.loadtxt(file_list[1], delimiter=' ')
print data.shape

neural_network_classify(data)

#benchmark(data2, 10)




# TDG:  14 of 75 are popular