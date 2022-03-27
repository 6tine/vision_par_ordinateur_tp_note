import dml
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from apprentissage_tests import getXandY,vectoriserTests

def main():
    #4.1.2 : Séparation par SVC
    classes_num = [0, 1]
    X, Y = getXandY(classes_num)
    classes = ['elephant', 'Leopards']
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel='poly', degree=3, C=1.0))
    clf.fit(X,Y)

    X_test = vectoriserTests()
    #X_tests[i][0] -> vecteurs à passer dans la fonction de prédiction
    # X_tests[i][1] -> nom de fichier
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
    print("---- taux de succès : ", (nbCorrect / nbPredict) * 100)
    print('résultat : ', result)
main()