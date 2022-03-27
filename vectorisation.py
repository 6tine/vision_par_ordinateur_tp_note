import cv2
import numpy as np
import pickle
import glob

def vectoriser(im, vocab):
    surf = cv2.xfeatures2d.SURF_create()
    #On extrait les SURF de l'image
    (kps, descs_tab) = surf.detectAndCompute(im,None)
    vector = np.zeros((len(vocab), len(vocab[0])))
    for i in range(len(descs_tab)):
        for j in range(len(descs_tab[i])):
            curr_desc = descs_tab[i][j]
            # On va calculer la distance entre le descripteur courant et chacun des mots du vocabulaire
            distances = np.absolute(vocab - curr_desc)
            # On va recupérer l'indice du mot le plus proche, donc en fonction de la distance minimale
            index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
            vector[index[0]][index[1]] += 1
    return vector

def vectoriserFromPaths(chemins, pickle_name):
    with open('matrice_vocabulaire_4_clusters.txt', 'r') as f:
        vocab = [[float(val) for val in line.split(' ')] for line in f]
    vocab_np = np.asarray(vocab)
    list_vector = []
    files_list = []
    i = 0
    for c in chemins:
        one_class_vectors = []
        one_class_files = []
        print(c)
        for file in glob.glob(c+"/*.jpg"):
            print('num : ', i)
            image = cv2.imread(file)
            vector = vectoriser(image, vocab_np)
            one_class_vectors.append(vector)
            one_class_files.append(file)
            i+=1
        list_vector.append(np.array(one_class_vectors))
        files_list.append(np.array(one_class_files))
    tab_final = []
    tab_final.append(files_list)
    tab_final.append(list_vector)
    print(tab_final)
    pickle.dump(tab_final, open(pickle_name, "wb"))
    return tab_final

def test():
    image_filename = './classes-caltech/elephant/image_0004.jpg'
    image = cv2.imread(image_filename)
    with open('matrice_vocabulaire_512_clusters.txt', 'r') as f:
        vocab = [[float(val) for val in line.split(' ')] for line in f]
    print(vocab)
    vector = vectoriser(image, vocab)
