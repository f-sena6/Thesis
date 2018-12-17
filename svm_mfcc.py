#########################################
#
#  Fred Sena
#  Thesis
#  Predict Popularity: SVM
#
#########################################


from sklearn.svm import SVC
from sklearn.utils import shuffle
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score


def get_pca(x):

    pca = PCA(n_components=0.99)   # Reduce by 99%
    x_pca = pca.fit_transform(x)

    print "PCA: ", x_pca.shape
    return x_pca

def benchmark(songs, count):

    mean_list = []

    for i in range(count):
        mean = svm_classify(songs)
        mean_list.append(mean)

    score = '%.2f%%' % (np.mean(mean_list)*100)

    #print mean_list
    print "\nAverage for {} runs: {}".format(count, score), '(+/- %.2f%%)' % (np.std(mean_list)*100)
    #print 


def svm_classify(data):

    x = data[:, 0:-1]
    y = data[:, -1]

    x, y = shuffle(x, y)

    #x = x/np.linalg.norm(x) # Normalize the data
    #x = preprocessing.scale(x)
    #x = get_pca(x)

    ###################
    # Set up the data #
    ###################
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    ##############
    # Set up SVM #
    ##############
    clf = SVC(max_iter=3000, kernel=kernels[0])

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
    print



    print "KFold Scores: ", cross_validation_score
    print "KFold Mean: %.2f%%" % (cross_validation_score.mean() * 100), '(+/- %.2f%%)' % (np.std(cross_validation_score)*100)

    return cross_validation_score.mean()



file_list = ['MFCC_Files/full_pca_metallica.txt', 'MFCC_Files/three_days_grace_pca.txt', 'MFCC_Files/three_days_grace_chromagram_pca.txt']
data = np.loadtxt(file_list[1], delimiter=' ')



print "Original Shape: ", data.shape


svm_classify(data)


#benchmark(data, 50)



