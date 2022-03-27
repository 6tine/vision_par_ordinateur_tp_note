from vocabulaire import vocabulaire, plotMetrics

def main():
    chemins = [
        './classes-caltech/elephant',
        './classes-caltech/Leopards',
        './classes-caltech/panda',
        './classes-caltech/wild_cat',
    ]
    #vocabulaire(4, chemins)
    #N_tab = [2,4,8,16,32,64,128,256,512,1024,2048,4096]
    N_tab = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    inerties_tab = []
    errors_tab = []
    for n in N_tab:
        inertie, error, vocab = vocabulaire(n, chemins)
        inerties_tab.append(inertie)
        errors_tab.append(error)
    plotMetrics(N_tab, inerties_tab, 1, 'Variance totale en fonction de N', 'var_tot_512_clusters.jpg')
    plotMetrics(N_tab, errors_tab, 2, 'Plus grande erreur en fonction de N', 'erreur_max_512_clusters.jpg')
    
main()