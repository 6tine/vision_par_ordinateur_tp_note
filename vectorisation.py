import cv2
from vocabulaire import vocabulaire

def vectoriser(im, vocab):
    descs_tab = []
    surf = cv2.xfeatures2d.SURF_create()
    #On extrait les SURF de l'image
    (kps, descs_tab) = surf.detectAndCompute(im,None)
    vector = [[0 for j in range(len(vocab[i]))] for i in range(len(vocab))]
    for i in range(len(descs_tab)):
        for j in range(len(descs_tab[i])):
            min_dist = float('inf')
            best_index = (0,0)
            #distances = np.abs(vocab - descs_tab[i][j])
            for k in range (len(vocab)):
                for l in range(len(vocab[k])):
                    curr_dist = abs(vocab[k][l] - descs_tab[i][j])
                    if(curr_dist < min_dist):
                        min_dist = curr_dist
                        best_index = (k,l)
            vector[best_index[0]][best_index[1]] += 1
    print('vector : ', vector)
    return vector

def main():
    image_filename = './classes-caltech/elephant/image_0004.jpg'
    image = cv2.imread(image_filename)
    with open('matrice_vocabulaire.txt', 'r') as f:
        vocab = [[float(val) for val in line.split(' ')] for line in f]
    print(vocab)
    vector = vectoriser(image, vocab)

main()