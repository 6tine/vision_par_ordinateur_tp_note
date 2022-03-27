from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from apprentissage_tests import getXandY,vectoriserTests
from sklearn import svm

def main():
    # 0 : elephant, 1 : leopards, 2 : panda, 3: chat sauvage
    X, Y = getXandY([0, 1, 2, 3])
    #svc_model = svm.SVC(kernel="poly", degree=1)
    #svc_model.fit(X, Y)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, Y)

    X_test = vectoriserTests()

    #result = svc_model.predict(X_test)
    result = clf.predict(X_test)

    print("résultat prédiction : ", result)

main()