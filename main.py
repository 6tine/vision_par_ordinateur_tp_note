import cv2
import glob
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def loadImages(chemins):
    images_classes_apprentissage = []
    images_classes_test = []
    for c in chemins:
        images = [cv2.imread(file) for file in glob.glob(c+"/*.jpg")]
        images_tests = [cv2.imread(file) for file in glob.glob(c+"/test/*.jpg")]
        images_classes_apprentissage.append(images)
        images_classes_test.append(images_tests)
    return images_classes_apprentissage, images_classes_test

def vocabulaire(N,chemins):
    # Load images
    images_classes_apprentissage, images_classes_test = loadImages(chemins)    

    # SURF
    descs_tab = []
    surf = cv2.xfeatures2d.SURF_create()
    for c in range(len(images_classes_apprentissage)):
        for image in images_classes_apprentissage[c]:
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (kps, descs) = surf.detectAndCompute(image,None)
            descs_tab.extend(np.array(descs))
    # Clusterisation
    kMeans = KMeans(n_clusters=N)
    kMeans.fit(np.array(descs_tab))
    centers = kMeans.cluster_centers_
    inertia = kMeans.inertia_
    np.savetxt('matrice_vocabulaire.txt',centers)
    error_max = np.amax(kMeans.transform(np.array(descs_tab)).sum(axis=1))
    print(error_max)
    return inertia, error_max

def plotMetrics(x, y, title):
    plt.plot(x,y)
    plt.title(title)
    plt.show()

def main():
    chemins = [
        './classes-caltech/elephant',
        './classes-caltech/Leopards',
        './classes-caltech/panda',
        './classes-caltech/wild_cat',
    ]
    vocabulaire(4, chemins)
    N_tab = [2,4,8,16,32,64,128,256,512,1024]
    inerties_tab = []
    errors_tab = []
    for n in N_tab:
        inertie, error = vocabulaire(n, chemins)
        inerties_tab.append(inertie)
        errors_tab.append(error)
    plotMetrics(N_tab, inerties_tab, 'Variance totale en fonction de N')
    plotMetrics(N_tab, errors_tab, 'Plus grande erreur en fonction de N')
main()