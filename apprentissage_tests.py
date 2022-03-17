import pickle
import numpy as np
import dml
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from vectorisation import vectoriserFromPaths

def getXandY(classes):
    matrice = pickle.load(open("base.pickle", "rb"))
    Y = []
    classes_filenames = matrice[0]
    classes_vectors = matrice[1]
    X = np.concatenate((classes_vectors[0], classes_vectors[1]))
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

def main():
    # on utilise la classe 0 : elephant et la classe 1 : leopards
    X, Y = getXandY([0, 1])
    s = dml.KDA(n_components=2, kernel='poly', degree=2)
    s.fit(X,Y)
    s.transform(X)
    print(s.transform())

    # Séparation par SVC
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X,Y)

    X_test = vectoriserTests()

    result = clf.predict(X_test)

    print("résultat prédiction : ", result)

main()