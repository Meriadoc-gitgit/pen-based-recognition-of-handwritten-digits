# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""


# Fonctions utiles
# Version de départ : Février 2024

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------ A COMPLETER
import random
# Version fit pour 10 labels allant de 0 -> 9
# ------------------------------------
def genere_train_test(desc_set, label_set, n_per_class):
    """Permet de générer une base d'apprentissage et une base de test.
    
    Args:
        desc_set (ndarray): Tableau avec des descriptions.
        label_set (ndarray): Tableau avec les labels correspondants (valeurs de 0 à 9).
        n_per_class (int): Nombre d'exemples par classe à mettre dans la base d'apprentissage.
        
    Returns:
        tuple: Un tuple contenant deux tuples, chacun avec la base d'apprentissage et la base de test sous la forme
               (data, labels).
    """
    nb_labels = len(np.unique(label_set))
    # Création des listes pour chaque classe
    train_data_by_class = [[] for _ in range(nb_labels)]
    test_data_by_class = [[] for _ in range(nb_labels)]
    
    # Séparation des données par classe
    for i in range(nb_labels):
        class_indices = np.where(label_set == i)[0]
        selected_indices = random.sample(class_indices.tolist(), n_per_class)
        for idx in class_indices:
            if idx in selected_indices:
                train_data_by_class[i].append(desc_set[idx])
            else:
                test_data_by_class[i].append(desc_set[idx])
    
    # Création des tableaux de données et de labels pour la base d'apprentissage
    train_data = np.concatenate([np.array(train_data_by_class[i]) for i in range(nb_labels)], axis=0)
    train_labels = np.concatenate([np.full(len(train_data_by_class[i]), i) for i in range(nb_labels)], axis=0)
    
    # Création des tableaux de données et de labels pour la base de test
    test_data = np.concatenate([np.array(test_data_by_class[i]) for i in range(nb_labels)], axis=0)
    test_labels = np.concatenate([np.full(len(test_data_by_class[i]), i) for i in range(nb_labels)], axis=0)
    
    return (train_data, train_labels), (test_data, test_labels)

    

# ------------------------ 
# genere_dataset_uniform
# ------------------------ 

def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    
    # COMPLETER ICI (remplacer la ligne suivante)
    data_desc = np.random.uniform(binf, bsup, (n*p, p))
    data_label = np.asarray(
    [-1 for i in range(0,n)] 
    + [+1 for i in range(0,n)])
    
    return data_desc, data_label
    #raise NotImplementedError("Please Implement this method")



# ------------------------ 
# genere_dataset_gaussian
# ------------------------ 

def genere_dataset_gaussian(
    positive_center, 
    positive_sigma, 
    negative_center, 
    negative_sigma, 
    nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    # COMPLETER ICI (remplacer la ligne suivante)
    class_moins1 = list(np.random.multivariate_normal(
        negative_center, 
        negative_sigma, 
        nb_points))
    
    class_1 = list(np.random.multivariate_normal(
        positive_center,
        positive_sigma,
        nb_points))
    
    fusion = class_moins1 + class_1
    labels = np.asarray(
    [-1 for i in range(0,nb_points)] 
    + [+1 for i in range(0,nb_points)])
    
    
    return np.array(fusion),labels
    
    #raise NotImplementedError("Please Implement this method")



# ------------------------ 
# plot2DSet
# ------------------------ 

def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    # COMPLETER ICI (remplacer la ligne suivante)
    
    # Extraction des exemples de classe -1:
    data_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data_positifs = desc[labels == +1]
    
    # Affichage de l'ensemble des exemples :
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1

    #raise NotImplementedError("Please Implement this method")




# ------------------------ 
# plot_frontiere
# ------------------------ 

def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])



# ------------------------ 
# create_XOR
# ------------------------ 

def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    negative_points_1 = np.random.multivariate_normal(
      np.array([0,0]), 
      np.array([[var,0],[0,var]]), 
      n
    )
    negative_points_2 = np.random.multivariate_normal(
      np.array([1,1]), 
      np.array([[var,0],[0,var]]), 
      n
    )
    positive_points_1 = np.random.multivariate_normal(
      np.array([1,0]), 
      np.array([[var,0],[0,var]]), 
      n
    )
    positive_points_2 = np.random.multivariate_normal(
      np.array([0,1]), 
      np.array([[var,0],[0,var]]), 
      n
    )
    
    desc = np.vstack(
        (negative_points_1, 
        negative_points_2, 
        positive_points_1, 
        positive_points_2)  
      )
    labels = np.asarray([-1 for i in range(2*n)] +\
        [+1 for i in range(2*n)])
    
    return desc, labels
    # raise NotImplementedError("Please Implement this method")


def normalisation(df) : 
  df_normalized = df.copy()

  # Pour chaque colonne du DataFrame
  for column in df_normalized.columns:
    # Récupérer le minimum et le maximum de la colonne
    min_val = df_normalized[column].min()
    max_val = df_normalized[column].max()
    
    # Normalisation Min-Max pour chaque valeur de la colonne
    df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)

  return df_normalized

def dist_euclidienne(exemple1, exemple2) : 
  exemple1 = np.array(exemple1)
  exemple2 = np.array(exemple2)

  s = 0
  
  for i in range(len(exemple1)) : 
    s += (exemple1[i] - exemple2[i])**2
  return np.sqrt(s)

def centroide(data):
    return np.mean(data, axis=0)

def dist_centroides(group1,group2) : 
  return dist_euclidienne(centroide(group1),centroide(group2))

def initialise_CHA(df) : 
  di = dict()
  for i in range(len(df)) : 
    di[i] = [i]

  return di


def fusionne(df, P0, verbose=False) : 
  P = copy.deepcopy(P0)
  di_e = []
  for i,_ in P.items() : 
    for j,_ in P.items() : 
      if i > j : 
        di_e.append((i,j,dist_centroides(df.iloc[P[i]],df.iloc[P[j]])))
      
  plus_proche = np.min([i for _,_,i in di_e])

  for i,j,k in di_e : 
    if k == plus_proche : 
      clef1 = j
      clef2 = i

  P[np.max(list(P.keys()))+1] = P[clef1] + P[clef2]
  P.pop(clef1)
  P.pop(clef2)
  
  if verbose : 
    print("fusionne: distance mininimale trouvée entre [",clef1,",",clef2,"] = ",plus_proche)
    print("fusionne: les 2 clusters dont les clés sont [",clef1,",",clef2,"] sont fusionnés")
    print("fusionne: on crée la  nouvelle clé ",len(P0)," dans le dictionnaire.")
    print("fusionne: les clés de [",clef1,",",clef2,"] sont supprimées car leurs clusters ont été fusionnés.")

  return P, clef1, clef2, plus_proche


def CHA_centroid(df, verbose=False, dendrogramme=False):
  depart = initialise_CHA(df)
  
  partition, clef1, clef2, dist = fusionne(df, depart)
  l = [[clef1, clef2, dist, len(depart[clef1]) + len(depart[clef2])]]
  
  if verbose : 
    print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")
    print("CHA_centroid: une fusion réalisée de  ",clef1," avec ", clef2," de distance  ",dist)
    print("CHA_centroid: le nouveau cluster contient  ",len(depart[clef1]) + len(depart[clef2]),"  exemples")
  while len(partition) > 1 : 
    tmp = copy.deepcopy(partition)
    partition, clef1, clef2, dist = fusionne(df, partition,verbose)
    l.append([clef1, clef2, dist, len(tmp[clef1]) + len(tmp[clef2])])
    if verbose : 
      print("CHA_centroid: une fusion réalisée de  ",clef1," avec ", clef2," de distance  ",dist)
      print("CHA_centroid: le nouveau cluster contient  ",len(tmp[clef1]) + len(tmp[clef2]),"  exemples")
  
  if dendrogramme : 
    # Paramètre de la fenêtre d'affichage: 
    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)    
    plt.xlabel("Indice d'exemple", fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme pour notre clustering :
    scipy.cluster.hierarchy.dendrogram(
        l, 
        leaf_font_size=24.,  # taille des caractères de l'axe des X
    )
  return l