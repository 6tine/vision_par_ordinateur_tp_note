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
    t = s.transform(X)
    print('s.transform(X)[1]')
    print(t[1])
    print('s.transform(X)[65]')
    print(t[65])
    print("debut s.tranform : ")
    print(t)
    print("fin s.tranform : ")
    m1 = statistics.mean(t[0:64][0])
    m2 = statistics.mean(t[65:264][0])
    print('moyenne m1 : ', m1)
    print('moyenne m2 : ', m2)

    print('degré = 4')
    s = dml.KDA(n_components=2, kernel='poly', degree=4)
    s.fit(X, Y)
    t = s.transform(X)
    print('s.transform(X)[1]')
    print(t[1])
    print('s.transform(X)[65]')
    print(t[65])
    print("debut s.tranform : ")
    print(t)
    print("fin s.tranform : ")
    m1 = statistics.mean(t[0:64][0])
    m2 = statistics.mean(t[65:264][0])
    print('moyenne m1 : ', m1)
    print('moyenne m2 : ', m2)

main()