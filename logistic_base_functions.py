import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


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
        X_train = X_train[:,element]
        X_test = X_test[:,element]
        nfeat = np.sum(element)
    else:
        n_feat = X_train.shape[1] #the total no of columns
    model = LogisticRegression(max_iter=1e9)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test,Y_pred)

    fitness = (
         alpha * np.log(acc)-
         - (1 - alpha) * (n_feat / np.sqrt(X_train.shape[1])) * acc * (1 + np.sin(golden_ratio))
    )
    return fitness,acc,n_feat

    

if __name__ =='__main__':
    # 
