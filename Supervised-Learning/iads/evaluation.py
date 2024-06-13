# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 




def crossval(X, Y, n_iterations, iteration):
    #############
    # A COMPLETER
    #############  

    index = np.random.permutation(len(X)) # mélange des index
    Xm = X[index]   
    Ym = Y[index]

    index_test = range(
        iteration * len(X) // n_iterations,
        (iteration + 1) * len(X) // n_iterations
    )
    index_train = [
        i for i in range(len(X)) \
            if i not in index_test
    ]
    
    return X[index_train], Y[index_train], X[index_test], Y[index_test]




# code de la validation croisée (version qui respecte la distribution des classes)

def crossval_strat(X, Y, n_iterations, iteration):
    #############
    # A COMPLETER
    ############# 

    #index = np.random.permutation(len(X)) # mélange des index
    #Xm = X[index]   
    #Ym = Y[index]   

    # Pour classe 1 
    X_1 = X[Y==1].copy()
    Y_1 = Y[Y==1].copy()

    index_test = range(
        iteration * len(X_1) // n_iterations,
        (iteration + 1) * len(X_1) // n_iterations
    )
    index_train = [
        i for i in range(len(X_1)) \
            if i not in index_test
    ]

    # Pour classe -1
    X_m1 = X[Y==-1].copy()
    Y_m1 = Y[Y==-1].copy()

    index_test = range(
        iteration * len(X_m1) // n_iterations,
        (iteration + 1) * len(X_m1) // n_iterations
    )
    index_train = [
        i for i in range(len(X_m1)) \
            if i not in index_test
    ]

    # Separation de dataset
    Xapp = np.array(
        list(X_m1[index_train]) + list(X_1[index_train])
    )
    Yapp = np.array(
        list(Y_m1[index_train]) + list(Y_1[index_train])
    )
    Xtest = np.array(
        list(X_m1[index_test]) + list(X_1[index_test])
    )
    Ytest = np.array(
        list(Y_m1[index_test]) + list(Y_1[index_test])
    )
    
    return Xapp, Yapp, Xtest, Ytest




# ------------------------ A COMPLETER
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L), np.sqrt(np.var(L))
    # raise NotImplementedError("Vous devez implémenter cette fonction !")    





import copy  # pour deepcopy()

def validation_croisee(C, DS, nb_iter, verbose=True):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    #############
    # A COMPLETER
    #############    
    Xm, Ym = DS
    perf = []
    
    for i in range(nb_iter) : 
        cl = copy.deepcopy(C)
        Xapp,Yapp,Xtest,Ytest = crossval(Xm, Ym, nb_iter, i)
        cl.train(Xapp,Yapp)
        perf.append(cl.accuracy(Xtest,Ytest))
        if verbose : 
            print("Itération",i," : taille base app.=",len(Xapp)," taille base test=",len(Xtest)," Taux de bonne classif:",cl.accuracy(Xtest, Ytest))
    
    mean, std = analyse_perfs(perf)
    return perf, mean, std
