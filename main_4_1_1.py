import dml
import statistics

from apprentissage_tests import getXandY

def main():
    # on utilise la classe 0 : elephant et la classe 1 : leopards
    classes_num = [0, 1]
    X, Y = getXandY(classes_num)
    #4.1.1 : Effet de la kernel-LDA
    print('degré = 2 ')
    s = dml.KDA(n_components=2, kernel='poly', degree=2)
    s.fit(X,Y)
    print("debut s.tranform : ")
    print(X)
    t = s.transform(X)
    print("fin s.tranform : ")
    print(t)
    print('s.transform(X)[1]')
    print(t[1])
    print('s.transform(X)[65]')
    print(t[65])
    #Pour l'éléphant il y a 65 image et pour le léopards il y en a 200
    t = t.reshape(len(t))
    m1 = statistics.mean(t[0:64])
    m2 = statistics.mean(t[65:264])
    v1 = statistics.variance(t[0:64])
    v2 = statistics.variance(t[65:264])
    print('moyenne m1 : ', m1)
    print('moyenne m2 : ', m2)
    print('variance v1 : ', v1)
    print('variance v2 : ', v2)

    print('degré = 4')
    s = dml.KDA(n_components=2, kernel='poly', degree=4)
    s.fit(X, Y)
    print("debut s.tranform : ")
    t = s.transform(X)
    print("fin s.tranform : ")
    print(t)
    print('s.transform(X)[1]')
    print(t[1])
    print('s.transform(X)[65]')
    print(t[65])
    t = t.reshape(len(t))
    m1 = statistics.mean(t[0:64])
    m2 = statistics.mean(t[65:264])
    v1 = statistics.variance(t[0:64])
    v2 = statistics.variance(t[65:264])
    print('moyenne m1 : ', m1)
    print('moyenne m2 : ', m2)
    print('variance v1 : ', v1)
    print('variance v2 : ', v2)

main()