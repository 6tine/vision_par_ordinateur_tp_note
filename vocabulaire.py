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
    np.savetxt('matrice_vocabulaire_512_clusters.txt',centers)
    distances = kMeans.transform(np.array(descs_tab))
    #Faire le maximum deux fois (essayer de récupérer un premier max sous forme de 
    #vecteur et faire un autre max dessus. Voir pour faire un reshape de kmeans.transform)
    #On veut la distance max pour le cluster correpondant au point, pas celle par rapport à tous les clusters
    error_max = np.amax(distances)
    print("N = ", N, "erreur max : ", error_max)
    return inertia, error_max, centers

def plotMetrics(x, y, i, title, filename):
    plt.figure(i)
    plt.plot(x,y)
    plt.title(title)
    plt.savefig(filename)
    plt.show()
