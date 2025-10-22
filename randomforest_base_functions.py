import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

# global hyperparameters 
alpha =0.7
max_iter = int(1e3/2)
g0=35
popsize = 50
# make random flips 
flip_prob = 0.04
RANDOM_SEED = 23

def initialize_population(popsize,n):
    """makes the poulation of popsize 
    each element is of n dimensions
    """
    return np.random.randint(0, 2, size=(popsize, n))

def fitness_function(X_train,Y_train,
                     element,
                     alpha =0.8,
                     golden_ratio = 1.618):
    
    """
    compute fitness of a single element for a logistic regression (classifier) 
    the element here is the feature map for the featrues to use
    """
    if element is not None:
        X_train_subset = X_train[:, element.astype(bool)]
        # X_test_subset = X_test[:, element.astype(bool)]
        n_feat = np.sum(element)
    else:
        X_train_subset = X_train
        n_feat = X_train.shape[1] #the total no of columns
    
    # model = RandomForestClassifier(n_estimators=75, max_depth=None, random_state=2)
    model = LogisticRegression(solver='liblinear', random_state=RANDOM_SEED)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    accuracies = cross_val_score(model, X_train_subset, Y_train, cv=skf, scoring='accuracy')
    acc = np.mean(accuracies) # Use the average accuracy

    acc = max(acc, 1e-10)
    # to punish the higher accuracies less changing log to raised to 2
    fitness = (
         alpha * np.log(acc)
         - (1 - alpha) * (n_feat / np.sqrt(X_train.shape[1]))  * (1 + np.sin(golden_ratio))
    )
    # fitness = alpha * np.log(acc) - (1 - alpha) * (n_feat / np.sqrt(X_train.shape[1])) * (1 + np.sin(golden_ratio))
    return fitness,acc,n_feat

def make_fitness_array(X_train,Y_train,population):
    """
    to return the fitness of the entire population """
    return np.array([fitness_function(X_train,Y_train,i) for i in population])

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
    # the 1e-10 added to make sure if fbest==fworst no division by 10 error
    Mi = np.array([abs(i - fbest) / ((fbest - fworst)+int(1e-10)) for i in fitnesses])
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
                             popsize =20, 
                             alpha =0.1,
                             max_iter=int(1e2),
                             g0=35):
    
    n = X_train.shape[1]
    population = initialize_population(popsize,n)
    # this returns tuple of [fitness,acc,nfeatures] for all the data points
    fitnesses_raw_data = make_fitness_array(X_train,Y_train,population)
    fitnesses = np.array([f[0] for f in fitnesses_raw_data])
    fitnesses = fitnesses.flatten() 
    fbest,fworst,best_idx, _ = best_worst_fitness(fitnesses)

    pbest = population.copy()
    pbest_fitness = fitnesses.copy()

    gbest = population[best_idx].copy()
    # to use in for loop 
    gbest_fitness = fbest

    for i in range(max_iter):
        fbest,fworst,_,_ = best_worst_fitness(fitnesses)
        Mi,Mbest,mi= computeMi_Mbest(fitnesses,fbest,fworst)

        G = compute_gravity_force(max_iter, i, g0, alpha)
        omega= compute_omega(i,max_iter)

        # compute mean best position weighted by mi
        mbest = np.sum(pbest * mi[:, np.newaxis], axis=0)

        acc = compute_acc(population,pbest,mbest,omega,mi)*1.2

        # doing the quantum gravaiton position update 
        # making some random data
        rand_phase = np.random.rand(*population.shape)
        # using gravtiation to change make the data converge
        new_population = population + G * acc * (2 * rand_phase - 1)
        # remvoign the data values which arent [0,1]
        new_population = np.clip(new_population, 0, 1)
        # make teh data a binary 0 or 1
        # changign the constant 0.5 to sigmoid
        sigmoid = 1 / (1 + np.exp(-new_population))
        new_population = (np.random.rand(*population.shape) > sigmoid).astype(int)


        # evaluating the fitness vales for this data
        new_fitness_raw = make_fitness_array(X_train, Y_train,  new_population)
        new_fitnesses= np.array([f[0]for f in new_fitness_raw])
        new_fitnesses = new_fitnesses.flatten() 
        
        rand_flip = np.random.rand(*new_population.shape) < flip_prob
        new_population[rand_flip] = 1 - new_population[rand_flip]


        # make the changes wherever the fitness gets better
        improved = new_fitnesses>pbest_fitness
        pbest[improved] = new_population[improved]
        pbest_fitness[improved] = new_fitnesses[improved]

        # to make a gbest update 
        new_gbest_idx = np.argmax(new_fitnesses)
        if new_fitnesses[new_gbest_idx] > gbest_fitness:
            gbest_fitness = new_fitnesses[new_gbest_idx]
            gbest = new_population[new_gbest_idx].copy()

        population=new_population
        fitnesses=new_fitnesses

        if i % 10 == 0 or i == max_iter - 1:
            print(f"Iter {i+1}/{max_iter} | Best Fitness: {gbest_fitness:.5f}")
    return gbest, gbest_fitness


if __name__ =='__main__':
    np.random.seed(RANDOM_SEED)
    current = os.getcwd()
    file_path = os.path.join(current, "data", "student-mat.csv")
    df = pd.read_csv(file_path, sep=';')  
    
    # separate features and target
    X = df.drop(columns=['G3'])
    y = (df['G3'] >= 10).astype(int)

    # keep the DataFrame after one-hot encoding
    X = pd.get_dummies(X)  
    scaler = StandardScaler()
    X[X.select_dtypes(include=np.number).columns] = scaler.fit_transform(X.select_dtypes(include=np.number))

    # save the column names
    feature_names = X.columns  

    # now convert to numpy array for QIGPSO
    X_values = X.values

    X_train, X_test, Y_train, Y_test = train_test_split(X_values, y, test_size=0.2,
                                                    random_state=11, stratify=y)

    best_features, best_fitness = qigpso_feature_selection(X_train, Y_train, popsize,alpha,max_iter,g0)
    
    # mpa indices to column names
    selected_feature_names = feature_names[np.where(best_features == 1)[0]]
    X_train_selected = X_train[:, best_features == 1]
    X_test_selected = X_test[:, best_features == 1]

    # print the results
    print("Best feature subset indices", np.where(best_features == 1)[0])
    print("Best feature names", selected_feature_names)
    print("Best fitness", best_fitness)

    # doing the classificaton
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED
    )
    model.fit(X_train_selected,Y_train)
    Y_pred = model.predict(X_test_selected)
    acc = accuracy_score(Y_test,Y_pred)

    print(acc)
