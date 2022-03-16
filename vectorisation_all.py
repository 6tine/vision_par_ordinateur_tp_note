from vectorisation import vectoriser
import glob
import pickle
import numpy as np
import cv2

def main():
    chemins = [
        './classes-caltech/elephant',
        './classes-caltech/Leopards',
        './classes-caltech/panda',
        './classes-caltech/wild_cat',
    ]
    with open('matrice_vocabulaire.txt', 'r') as f:
        vocab = [[float(val) for val in line.split(' ')] for line in f]
    list_vector = []
    for c in chemins:
        for file in glob.glob(c+"/*.jpg"):
            image = cv2.imread(file)
            vector = vectoriser(image, vocab)
            list_vector.extend(np.array(vector))
    print("list_vector : ", list_vector)
    pickle.dump(np.array(list_vector), open("base.pickle", "wb"))

main()