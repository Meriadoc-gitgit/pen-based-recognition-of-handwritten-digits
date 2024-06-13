# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy
# ------------------------ 

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

import copy 
def fusionne(df, P0, verbose=False,linkage='centroid') : 
  P = copy.deepcopy(P0)
  di_e = []
  for i,_ in P.items() : 
    for j,_ in P.items() : 
      if i > j : 
        if linkage == 'centroid' : 
          di_e.append((i,j,dist_centroides(df.iloc[P[i]],df.iloc[P[j]])))
        elif linkage == 'complete' : 
          di_e.append((i,j,dist_complete(df.iloc[P[i]],df.iloc[P[j]])[0]))
        elif linkage == 'simple' : 
          di_e.append((i,j,dist_simple(df.iloc[P[i]],df.iloc[P[j]])[0]))
        elif linkage == 'average' : 
          di_e.append((i,j,dist_average(df.iloc[P[i]],df.iloc[P[j]])[0]))
      
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


def dist_complete(group1,group2) :
  max_dist = float('-inf')
  for i in range(len(group1)) : 
    for j in range(len(group2)) : 
      dist = dist_euclidienne(group1.iloc[i], group2.iloc[j])  # Remplacez par votre fonction de distance
      if dist > max_dist:
        max_dist = dist
        indices = (j, i)
  return max_dist, indices

def dist_simple(group1,group2) :
  min_dist = float('inf')
  for i in range(len(group1)) : 
    for j in range(len(group2)) : 
      dist = dist_euclidienne(group1.iloc[i], group2.iloc[j])  # Remplacez par votre fonction de distance
      if dist < min_dist:
        min_dist = dist
        indices = (j, i)
  return min_dist, indices

def dist_average(group1,group2) : 
  total_dist = 0
  count = 0
  for i in range(len(group1)) : 
    for j in range(len(group2)) :
      total_dist += dist_euclidienne(group1.iloc[i], group2.iloc[j])  # Remplacez par votre fonction de distance
      count += 1
  avg_dist = total_dist / count
  new_size = len(group1) + len(group2)
  return avg_dist, new_size


def CHA(df,linkage='centroid', verbose=False,dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    ############################ A COMPLETER
    depart = initialise_CHA(df)
  
    partition, clef1, clef2, dist = fusionne(df, depart,linkage)
    l = [[clef1, clef2, dist, len(depart[clef1]) + len(depart[clef2])]]
    
    if verbose : 
        print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")
        print("CHA_centroid: une fusion réalisée de  ",clef1," avec ", clef2," de distance  ",dist)
        print("CHA_centroid: le nouveau cluster contient  ",len(depart[clef1]) + len(depart[clef2]),"  exemples")
    while len(partition) > 1 : 
        tmp = copy.deepcopy(partition)
        partition, clef1, clef2, dist = fusionne(df, partition,verbose,linkage)
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
        plt.grid()
    return l

    

"""
################################################################################
"""
def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """

    ############# A COMPLETER 
    # Calcul du centroïde
    c = centroide(Ens)
    
    # Calcul des distances au carré entre chaque point et le centroïde
    distances_carre = np.sum((Ens - c)**2, axis=1)
    
    # Somme des distances au carré
    inertie = np.sum(distances_carre)
    
    return inertie


import random

def init_kmeans(K,Ens):
    """ int * Array -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: Array contenant n exemples
    """

    ############# A COMPLETER 
    return np.array(Ens.sample(n=K))

def plus_proche(Exe,Centres):
    """ Array * Array -> int
        Exe : Array contenant un exemple
        Centres : Array contenant les K centres
    """

    ############# A COMPLETER 
    distances = [clust.dist_euclidienne(Exe, centroide) for centroide in Centres]
    
    return np.argmin(distances)


def affecte_cluster(Base,Centres):
    """ Array * Array -> dict[int,list[int]]
        Base: Array contenant la base d'apprentissage
        Centres : Array contenant des centroides
    """
    
    ############# A COMPLETER 
    di = dict()
    
    # Initialisation
    for i in range(len(Centres)) : 
        di[i] = []
    for i in range(len(Base)) : 
        pproche = plus_proche(Base.iloc[i],Centres)
        di[pproche].append(i)
    return di

import copy
def nouveaux_centroides(Base,U):
    """ Array * dict[int,list[int]] -> DataFrame
        Base : Array contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    
    ############# A COMPLETER 
    l = []
    for k,v in U.items() : 
        l.append(list(clust.centroide(Base.iloc[U[k]])))
    return np.array(l)

def inertie_globale(Base, U):
    """ Array * dict[int,list[int]] -> float
        Base : Array pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    
    ############# A COMPLETER 
    inertie = 0
    for indice_centroide, indices_exemples in U.items():
        exemples_cluster = Base.iloc[indices_exemples]
        inertie += inertie_cluster(exemples_cluster)
    return inertie

def kmoyennes(K, Base, epsilon, iter_max):
    """ int * Array * float * int -> tuple(Array, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : Array pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    
    ############# A COMPLETER 
    
    if K<=1 : 
        print("Error : K > 1")
        return [],[]

    # Initialisation aléatoire des centroides - 1
    Centres = init_kmeans(K, Base)
    inertie_prec = 0
    # 2
    for i in range(iter_max):
        U = affecte_cluster(Base, Centres)
        new_centers = nouveaux_centroides(Base, U)
        inertie = inertie_globale(Base, U)
        diff = abs(inertie - inertie_prec)
        
        # Mise à jour des centroides et de l'inertie précédente
        Centres = new_centers
        inertie_prec = inertie

        if i > 0 : 
            print("Iteration", i, "Inertie :", inertie, "Différence :", diff)
            # Vérification de la convergence
            if diff < epsilon:
                break

    return Centres, U

import matplotlib.cm as cm

def affiche_resultat(Base,Centres,Affect):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """

    ############# A COMPLETER 
    couleurs = cm.tab20(np.linspace(0, 1, len(Affect)))
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
    
    for k,v in Affect.items() : 
        b_v = np.array(Base.iloc[Affect[k]])
        plt.scatter(b_v[:,0],b_v[:,1],color=couleurs[k])