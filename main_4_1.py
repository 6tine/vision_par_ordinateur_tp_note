import dml
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from apprentissage_tests import getXandY,vectoriserTests

def main():
    # on utilise la classe 0 : elephant et la classe 1 : leopards
    X, Y = getXandY([0, 1])
    #4.1.1 : Effet de la kernel-LDA
    s = dml.KDA(n_components=2, kernel='poly', degree=2)
    s.fit(X,Y)
    s.transform(X)
    print("debut s.tranform : ")
    print(s.transform())
    print("fin s.tranform : ")

    #4.1.2 : Séparation par SVC
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X,Y)

    X_test = vectoriserTests()

    result = clf.predict(X_test)

    print("résultat prédiction : ", result)

main()