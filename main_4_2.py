from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from apprentissage_tests import getXandY,vectoriserTests
from sklearn import svm
import numpy as np

def testSVC():
    # 0 : elephant, 1 : leopards, 2 : panda, 3: wild cat
    classes_num = [0, 1, 2, 3]
    X, Y = getXandY(classes_num)
    classes = ['elephant', 'Leopards','panda','wild_cat']
    #svc_model = svm.SVC(kernel="poly", degree=2)
    #svc_model.fit(X, Y)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='poly'))
    clf.fit(X, Y)

    X_test = vectoriserTests()
    np.random.shuffle(X_test)

    #result = svc_model.predict(X_test)

    d = [col[0] for col in X_test[:][:][:]]
    files = [col[1][0] for col in X_test[:][:][:]]
    result = clf.predict(d)
    nbCorrect = 0
    nbPredict = len(result)
    for i in range(nbPredict):
        correct = classes[result[i]] in files[i]
        if correct:
            nbCorrect += 1
        print('---- image : ', files[i], '---- prediction : ', classes[result[i]],
              '---- correct : ', correct)
    print("---- Nombre de succès : ", nbCorrect)
    print("---- Nombre d'erreur : ", nbPredict - nbCorrect)
    print("---- taux de succès : ", (nbCorrect/nbPredict)*100)

if __name__ == '__main__':
    svc_model = svm.SVC(kernel="poly", degree=2)
    testSVC()