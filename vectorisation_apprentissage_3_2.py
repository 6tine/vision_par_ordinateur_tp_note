from vectorisation import vectoriserFromPaths

chemins = [
        './classes-caltech/elephant',
        './classes-caltech/Leopards',
        './classes-caltech/panda',
        './classes-caltech/wild_cat',
    ]
vectoriserFromPaths(chemins, "base.pickle")