import pickle
import numpy as np

from vectorisation import vectoriserFromPaths

def getXandY(classes):
    matrice = pickle.load(open("base_512_clusters.pickle", "rb"))
    Y = []
    classes_filenames = matrice[0]
    classes_vectors = matrice[1]
    t = tuple(classes_vectors[c] for c in classes)
    X = np.concatenate(t)
    # Chaque ligne de X et Y est associée à une image d'une classe
    # X indique les vecteurs de chaque images
    # Y indique à quelle classe appartient chaque image (chaque ligne de X)
    for c in classes:
        tmp = np.full(len(classes_filenames[c]), c)
        Y.extend(tmp)
    Y = np.array(Y)
    X = X.reshape(len(Y),-1)
    return X,Y

def vectoriserTests():
    chemins = [
        './classes-caltech/elephant/test',
        './classes-caltech/Leopards/test',
        './classes-caltech/panda/test',
        './classes-caltech/wild_cat/test',
    ]
    matrice = vectoriserFromPaths(chemins, "tests.pickle")
    classes_vectors = matrice[1]
    X_test = np.concatenate((classes_vectors[0], classes_vectors[1], classes_vectors[2], classes_vectors[3]))
    X_test = X_test.reshape(len(X_test), -1)
    return X_test