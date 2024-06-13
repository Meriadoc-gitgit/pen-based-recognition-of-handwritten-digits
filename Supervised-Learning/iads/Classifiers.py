# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2024

# Import de packages externes
import numpy as np
import pandas as pd

# ---------------------------
# Classifieur
# ---------------------------

class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        
        #raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        pass
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        pass
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        pass
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # ------------------------------
        # COMPLETER CETTE FONCTION ICI : 
        
        # ............
        
        # ------------------------------
        """pred = []
        print(len(desc_set))
        for i in range(len(desc_set)) : 
            print(i)
            pred.append(self.predict(desc_set[i]))"""
        pred = [self.predict(desc_set[i]) for i in range(len(desc_set))]
        
        good_rate = 0
        for i in range(len(desc_set)) : 
            if pred[i] == label_set[i] :
                good_rate += 1
        return good_rate / len(desc_set)



# ---------------------------
# ClassifieurKNN
# ---------------------------
# Version adjustée de ClassifierKNN pour 10 labels rangées de 0 -> 9
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k):
        """ Constructeur de ClassifierKNN
            Argument:
                - input_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.k = k

    def score(self, x):
        """ Rend la proportion des labels parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distances = np.sqrt(np.sum((self.desc_set - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.label_set[nearest_indices]
        label_counts = np.bincount(nearest_labels, minlength=10)
        return np.argmax(label_counts)

    def predict(self, x):
        """ Rend la prédiction sur x (label de 0 à 9)
            x: une description : un ndarray
        """
        return self.score(x)

    def train(self, desc_set, label_set):
        """ Permet d'entraîner le modèle sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set




# ---------------------------
# ClassifierLineaireRandom
# ---------------------------

class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.w = np.random.uniform(-1, 1, input_dimension)
        
        #raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
        print("Pas d'apprentissage pour ce classifieur !")
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
        #raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) > 0 else -1
        #raise NotImplementedError("Please Implement this method")
    



# ---------------------------
# ClassifierPerceptron
# ---------------------------
# Version adjustée pour 10 labels allant de 0->9
# ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, num_labels=10, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - num_labels (int): nombre de classes dans la classification
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        super().__init__(input_dimension)
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        # --- Initialisation de w ---
        self.w = np.zeros((num_labels, input_dimension))  # Each label gets its own weight vector
        if not init:
            for i in range(num_labels):
                for j in range(input_dimension):
                    self.w[i, j] = ((2*np.random.random()-1) * 0.001)

        self.allw = [self.w.copy()]  # stockage des premiers poids

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """   
        # Choisir aléatoirement un exemple x_i de X
        data = list(zip(desc_set, label_set))
        np.random.shuffle(data)

        for x_i, y_i in data:
            predict = self.predict(x_i)
            if y_i != predict:
                self.w[y_i] += self.learning_rate * x_i
                self.w[predict] -= self.learning_rate * x_i

            self.allw.append(self.w.copy())
        
        return self.w 

    def score(self, x):
        """ rend le score de prédiction sur x (un vecteur de scores pour chaque classe)
            x: une description
        """
        return np.dot(self.w, x)

    def predict(self, x):
        """ rend la prediction sur x (un chiffre de 0 à 9)
            x: une description
        """
        scores = self.score(x)
        return np.argmax(scores)

    def get_allw(self):
        return self.allw

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """  
        norm_diff_values = []      
        i = 0
        
        while i < nb_max: 
            w_old = self.w.copy()
            self.w = self.train_step(desc_set, label_set)
            
            norm_diff = np.linalg.norm(w_old - self.w)
            norm_diff_values.append(norm_diff)
            
            if norm_diff < seuil:
                break
            
            i += 1
        
        return norm_diff_values





# ---------------------------
# ClassifierPerceptronBiais
# ---------------------------

# Remarque : quand vous transférerez cette classe dans le fichier classifieur.py 
# de votre librairie, il faudra enlever "classif." en préfixe de la classe ClassifierPerceptron:

class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate=learning_rate, init=init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        ##################
        ### A COMPLETER !
        ##################
        # Ne pas oublier d'ajouter les poids à allw avant de terminer la méthode
        self.desc_set = desc_set
        self.label_set = label_set
        data = np.array(
            [(i,desc_set[i],label_set[i]) \
                for i in range(len(desc_set))],dtype=object
            )
        
        np.random.shuffle(data)
        for _, x_i, y_i in data : 
            f_x_i = self.score(x_i)

            if f_x_i * y_i < 1 : 
                self.w += self.learning_rate * (y_i - f_x_i) * x_i
            
            self.allw.append(self.w.copy())
        return self.w
        # raise NotImplementedError("Vous devez implémenter cette méthode !")    
# ------------------------ 






# Donner la définition de la classe ClassifierMultiOAA

# Vous pouvez avoir besoin d'utiliser la fonction deepcopy de la librairie standard copy:
import copy 


# ------------------------ A COMPLETER :

class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """

        self.cl_bin = cl_bin 
        self.classifieurs = []

        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """      
        self.classifiers = []
        for c in np.unique(label_set):
            cl = copy.deepcopy(self.cl_bin)
            labels = np.where(label_set == c, 1, -1)
            cl.train(desc_set, labels)
            self.classifiers.append(cl)

        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        scores = [clf.score(x) for clf in self.classifiers]
        return scores

        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        scores = self.score(x)
        return np.argmax(scores)
        
        # raise NotImplementedError("Vous devez implémenter cette fonction !")









# CLasse (abstraite) pour représenter des noyaux
class Kernel():
    """ Classe pour représenter des fonctions noyau
    """
    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim
    
    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """        
        raise NotImplementedError("Please Implement this method")



class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """
    def __init__(self):
        """ Constructeur de KernelBias
            pas d'argument, les dimensions sont figées
        """
        # Appel du constructeur de la classe mère
        super().__init__(2,3)
        
    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3            
            rajoute une 3e dimension au vecteur donné
        """
        
        if (V.ndim == 1): # on regarde si c'est un vecteur ou une matrice
            W = np.array([V]) # conversion en matrice
            V_proj = np.append(W,np.ones((len(W),1)),axis=1)
            V_proj = V_proj[0]  # on rend quelque chose de la même dimension
        else:
            V_proj = np.append(V,np.ones((len(V),1)),axis=1)
            
        return V_proj
        



# ------------------------ A COMPLETER :

class KernelPoly(Kernel):
    def __init__(self):
        """ Constructeur de KernelPoly
            pas d'argument, les dimensions sont figées
        """
        # Appel du constructeur de la classe mère
        super().__init__(2,6)
        
    def transform(self,V):
        """ ndarray de dim 2 -> ndarray de dim 6            
            transforme un vecteur 2D en un vecteur 6D de la forme (1, x1, x2, x1*x1, x2*x2, x1*x2)
        """
        ## TODO
        if (V.ndim == 1) : 
          W = np.array([V]) # conversion en matrice
          V_proj = np.append(np.ones((len(W),1)),W,axis=1)
          x1, x2 = V[0], V[1]

          
          V_proj = V_proj[0]  # on rend quelque chose de la même dimension
          V_proj = np.append(V_proj, x1*x1)
          V_proj = np.append(V_proj, x2*x2)
          V_proj = np.append(V_proj, x1*x2)
        
        else : 
          V_proj = np.append(np.ones((len(V),1)),V,axis=1)
          x1, x2 = V[:,0], V[:,1]
          
          one = np.ones((len(V),1))
          x11 = x1 * x1
          x22 = x2 * x2
          x12 = x1 * x2

          V_proj = np.append(V_proj, [one[i] * x11[i] for i in range(len(one))],axis=1)
          V_proj = np.append(V_proj, [one[i] * x22[i] for i in range(len(one))],axis=1)
          V_proj = np.append(V_proj, [one[i] * x12[i] for i in range(len(one))],axis=1)

        return V_proj
        # raise NotImplementedError("Please Implement this method")





# ------------------------ A COMPLETER :
class ClassifierPerceptronKernel(ClassifierPerceptron):
    """ Perceptron de Rosenblatt kernelisé
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        super().__init__(input_dimension, learning_rate, bool(init))
        self.kernel = noyau

        # --- Initialisation de w ---
        self.w = np.zeros(self.kernel.get_output_dim())
        if not init :
            for i in range(self.kernel.get_output_dim()):
                self.w[i] = ((2*np.random.random()-1) * 0.001)

        # self.allw =[self.w.copy()] # stockage des premiers poids
        
        # raise NotImplementedError("Please Implement this method")
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        # Mélanger les indices des exemples
        data = np.array([(i,desc_set[i],label_set[i]) for i in range(len(desc_set))],dtype=object)
        np.random.shuffle(data)

        for _,x_i,y_i in data :
            predict = self.predict(x_i)

            if y_i != predict : 
                self.w += self.learning_rate * y_i * self.kernel.transform(x_i)

            #self.allw.append(self.w.copy())
        
        return self.w 

        #raise NotImplementedError("Please Implement this method")
     
    def score(self,x):
        """ rend le score de prédiction sur x 
            x: une description (dans l'espace originel)
        """
        # Projeter la description dans l'espace de dimension supérieure
        x_transformed = self.kernel.transform(x)
        # Calculer le score de prédiction
        score = np.dot(self.w, x_transformed)
        # Appliquer la fonction d'activation
        return score
        #raise NotImplementedError("Please Implement this method")
    





# code de la classe pour le classifieur ADALINE


# ATTENTION: contrairement à la classe ClassifierPerceptron, on n'utilise pas de méthode train_step()
# dans ce classifier, tout se fera dans train()


#TODO: Classe à Compléter

class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.learning_rate = learning_rate
        self.history = history
        self.niter_max = niter_max

        # Initialisation de w aléatoire
        self.w = np.random.randn(input_dimension) * learning_rate
        
        self.allw = [] # stockage des premiers poids
        # raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        # BOUCLE
        self.desc_set = desc_set
        self.label_set = label_set
        
        if self.history : 
            self.allw.append(self.w.copy())
        
        lastw = self.w + 1

        for i in range(self.niter_max) : 
            index = int(np.random.rand() * len(desc_set))
            x,y = desc_set[index], label_set[index]

            yhat = self.score(x)

            self.w -= self.learning_rate * x * (self.score(x) - y)

            if self.history : 
                self.allw.append(self.w.copy())

            if i%len(desc_set) == 0 : 
                if np.max(np.abs(lastw - self.w)) < 1e-3 : 
                    print("cvg in"+str(i)+"iterations")
                    return
                lastw = self.w

        # raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)

        # raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))

        # raise NotImplementedError("Please Implement this method")




# code de la classe ADALINE Analytique

class ClassifierADALINE2(Classifier):
# ...... COMPLETER
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)

        # Initialisation de w aléatoire
        # self.w = np.random.randn(input_dimension) * learning_rate
        
        # self.allw = [] # stockage des premiers poids
        # raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.w = np.linalg.solve(desc_set.T @ desc_set, desc_set.T @ label_set)

        # raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)

        # raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))

        # raise NotImplementedError("Please Implement this method")



def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    ########################## COMPLETER ICI 
    valeurs, nb_fois = np.unique(Y, return_counts=True)

    return valeurs[np.argmax(nb_fois)]
    
    ##########################
        

import math
def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """
    ########################## COMPLETER ICI 
    s = 0.
    for i in range(len(P)) : 
        if P[i] != 0 and len(P) > 1 : 
            s -= P[i] * math.log(P[i]) / math.log(len(P))
    
    return s
    ##########################

def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    ########################## COMPLETER ICI 
    _, nb_fois = np.unique(Y, return_counts=True)
    
    l = [i / len(Y) for i in nb_fois]
    return shannon(l)
    ##########################


# La librairie suivante est nécessaire pour l'affichage graphique de l'arbre:
import graphviz as gv

# Eventuellement, il peut être nécessaire d'installer graphviz sur votre compte:
# pip install --user --install-option="--prefix=" -U graphviz

class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple 
            on rend la valeur None si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return None
    
    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        nb_feuilles = 0
        if not self.Les_fils : 
            return nb_feuilles
        for fils in self.Les_fils : 
            if not self.Les_fils[fils].Les_fils : 
                nb_feuilles += 1
            else : 
                nb_feuilles += self.Les_fils[fils].compte_feuilles()
        return nb_feuilles
        # raise NotImplementedError("A implémenter plus tard (voir plus loin)")
     
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g

def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    entropie_ens = entropie(Y)                        # 1
    if (entropie_ens <= epsilon):                     # 2
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:                                             # 3
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
        for j in range(len(X[0])) : 
            entropie_attribut = 0.  # Initialisation de l'entropie de l'attribut
            valeurs_attribut = np.unique(X[:, j])  # Obtenir les valeurs uniques pour l'attribut j
            for v in valeurs_attribut:
                indices = np.where(X[:, j] == v)  # Indices où l'attribut j a la valeur v
                Y_subset = Y[indices]  # Sous-ensemble des étiquettes correspondant à ces indices
                poids = len(Y_subset) / len(Y)  # Calcul du poids de cette valeur de l'attribut
                entropie_attribut += poids * entropie(Y_subset)  # Calcul de l'entropie conditionnelle
            #print(entropie_attribut, nom_dataset.columns[i_best])
            # Mise à jour de l'attribut qui minimise l'entropie
            if entropie_attribut < min_entropie:
                min_entropie = entropie_attribut
                i_best = j
                Xbest_valeurs = valeurs_attribut

        #############################################

        if (entropie_ens - min_entropie) == 0: # pas de gain d'information possible
            # ARRET : on crée une feuille
            noeud = NoeudCategoriel(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
            
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud
    

class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ##################
        ## COMPLETER ICI !
        ##################
        noms = [nom for nom in self.LNoms if nom != 'class']
        self.racine = construit_AD(desc_set, label_set, self.epsilon,noms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        ##################
        ## COMPLETER ICI !
        ##################
        return self.racine.classifie(x)
        

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()
    
    def draw(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)


def partitionne(m_desc,m_class,n,s):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - n : (int) numéro de colonne de m_desc
            - s : (float) seuil pour le critère d'arrêt
        Hypothèse: m_desc peut être partitionné ! (il contient au moins 2 valeurs différentes)
        output: un tuple composé de 2 tuples
    """
    return ((m_desc[m_desc[:,n]<=s], m_class[m_desc[:,n]<=s]), \
            (m_desc[m_desc[:,n]>s], m_class[m_desc[:,n]>s]))


import graphviz as gv

class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        #############
        # COMPLETER CETTE PARTIE 
        #
        #############
        if self.est_feuille() : 
            return self.classe
        if exemple[self.attribut] <= self.seuil : 
            return self.Les_fils["inf"].classifie(exemple)
        else : 
            return self.Les_fils["sup"].classifie(exemple)
        
        # raise NotImplementedError("A implémenter plus tard (voir plus loin)")

    
    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        #############
        # COMPLETER CETTE PARTIE AUSSI
        #
        #############
        nb_feuilles = 0
        if not self.Les_fils : 
            return nb_feuilles
        for fils in self.Les_fils : 
            if not self.Les_fils[fils].Les_fils : 
                nb_feuilles += 1
            else : 
                nb_feuilles += self.Les_fils[fils].compte_feuilles()
        return nb_feuilles
        
        # raise NotImplementedError("A implémenter plus tard (voir plus loin)")
     
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g



def construit_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape

    _, nb_fois = np.unique(Y, return_counts=True)
    
    l = [i / len(Y) for i in nb_fois]
    
    entropie_classe = shannon(l)    # entropie(Y), on effectue des instructions facultatives vu que entropie(Y) retourne un problème incompréhensible de définition de fonction 
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = 0.0  # meilleur gain trouvé (initalisé à 0.0 => aucun gain)
        i_best = -1     # numéro du meilleur attribut (init à -1 (aucun))
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible), dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))       CHÚ Ý !! CHIA PHẦN
        for i in range(len(X[0])) : 
            (seuil_trouve, entropie), liste_vals = discretise(X, Y, i)
            
            if seuil_trouve is not None:  # Si on a trouvé un seuil
                gain_info = entropie_classe - entropie  # Calcul du gain d'information
                
                # Vérifier si ce gain est le meilleur trouvé jusqu'à présent
                if gain_info > gain_max:
                    gain_max = gain_info
                    i_best = i
                    Xbest_tuple = partitionne(X, Y, i_best, seuil_trouve)
                    Xbest_seuil = seuil_trouve
            

        
        
        ############
        if (i_best != -1): # Un attribut qui amène un gain d'information >0 a été trouvé
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)
                
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil, \
                              construit_AD_num(left_data,left_class, epsilon, LNoms), \
                              construit_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud

class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()
    
    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
# ---------------------------

