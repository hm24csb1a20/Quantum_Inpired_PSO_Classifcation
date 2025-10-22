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
    """return the best and the worst perfoming element fitness values and their indexes"""
    fbest = np.max(fitnesses)
    fworst = np.min(fitnesses)
    best_idx = np.argmax(fitnesses)
    worst_idx = np.argmin(fitnesses)
    return fbest, fworst, best_idx, worst_idx

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
    population = initialize_population(popsize)
    n = X_train.shape[1]
    # this returns tuple of [fitness,acc,nfeatures] for all the data points
    fitnesses_raw_data = make_fitness_array(X_train,Y_train,X_test, Y_test,population)
    fitnesses = np.array(f[0] for f in fitnesses_raw_data)
    fbest,fworst,best_idx, _ = best_worst_fitness(fitnesses)

    pbest = population.copy()
    pbest_fitness = fitnesses.copy()

    gbest = population[best_idx].copy()
    # to use in for loop 
    gbest_fitness = fbest

    for i in range(max_iter):
        fbest,fworst,_,_ = best_worst_fitness(fitnesses)
        Mi,Mbest,mi= computeMi_Mbest(max_iter,i,g0,alpha)

        G = compute_gravity_force(max_iter,i,g0,alpha)
        omega= compute_omega(i,max_iter)

        # compute mean best position weighted by mi
        mbest = np.sum(pbest * mi[:, np.newaxis], axis=0)

        acc = compute_acc(population,pbest,mbest,omega)

        # doing the quantum gravaiton position update 
        # making some random data
        rand_phase = np.random.rand(*population.shape)
        # using gravtiation to change make the data converge
        new_population = population + G * acc * (2 * rand_phase - 1)
        # remvoign the data values which arent [0,1]
        new_population = np.clip(new_population, 0, 1)
        # make teh data a binary 0 or 1
        new_population = (new_population > 0.5).astype(int)





if __name__ =='__main__':
    # 
