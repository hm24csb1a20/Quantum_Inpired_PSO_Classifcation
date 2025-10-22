import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# global hyperparameters 
alpha =0.8
max_iter = 1e9
g0=9.8

def initialize_population(popsize,n):
    """makes the poulation of popsize 
    each element is of n dimensions
    """
    return np.random.randint(0, 2, size=(popsize, n))

def fitness_function(X_train,Y_train,
                     X_test, Y_test,
                     element,
                     alpha =0.8,
                     golden_ratio = 1.618):
    
    """
    compute fitness of a single element for a logistic regression (classifier) 
    the element here is the feature map for the featrues to use
    """
    if(element!=None):
        X_train = X_train[:, element.astype(bool)]
        X_test = X_test[:, element.astype(bool)]
        n_feat = np.sum(element)
    else:
        n_feat = X_train.shape[1] #the total no of columns
    model = LogisticRegression(max_iter=int(1e9))
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test,Y_pred)
    acc = max(acc, 1e-10)  # prevent log(0)
    fitness = (
         alpha * np.log(acc)
         - (1 - alpha) * (n_feat / np.sqrt(X_train.shape[1])) * acc * (1 + np.sin(golden_ratio))
    )
    return fitness,acc,n_feat

def make_fitness_array(X_train,Y_train,X_test, Y_test,population):
    """
    to return the fitness of the entire population """
    return np.array([fitness_function(X_train,Y_train,X_test, Y_test,i) for i in population])

def best_worst_fitness(fitnesses):
    """return the best and the worst perfoming element fitness values"""
    return max(fitnesses),min(fitnesses)

def computeMi_Mbest(fitnesses,fbest,fworst):
    """returns the array of masses of each elemnet
    the best Mi 
    the best mi (normalized Mi)"""
    Mi = np.array([abs(i - fbest) / (fbest - fworst) for i in fitnesses])
    mi = Mi / np.sum(Mi)
    return Mi,max(Mi),mi

def compute_omega(iteration, max_iter, omega_max=1.0, omega_min=0.4):
    """linearly decaying omega (inertia/contraction factor)"""
    return omega_max - (omega_max - omega_min) * (iteration / max_iter)

def compute_acc(population, pbest, mbest, omega, mi):
    r1 = np.random.rand(*population.shape)
    r2 = np.random.rand(*population.shape)

    term1 = omega * r1 * (pbest - population)
    term2 = (1 - omega) * r2 * (mbest - population)

    acceleration = (term1 + term2) * mi[:, np.newaxis]  # scale by normalized mass
    return acceleration

def compute_distance_matrix(population):
    """computer pairwise distance matrix between all particles."""
    diff = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    # just some fancy math to calcuate euclidean distance 
    R = np.linalg.norm(diff, axis=2)
    return R

def compute_gravity_force(max_iter,
                           i ,
                          g0 = 9.8,
                          alpha =0.8):
    return g0*np.exp(-1*alpha*i/max_iter)

def qigpso_feature_selection(X_train,Y_train,
                             X_test,Y_test,
                             popsize =20, 
                             alpha =0.8,
                             max_iter=55,
                             g0=9.8):
    populationo = initialize_population(popsize)
    fintesses = make_fitness_array()

if __name__ =='__main__':
    # 
