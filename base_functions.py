import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def intialize_population(rangest,rangeend,n):
    """
    making the population of 
    [rangest,rangeend] range 
    and of n entries 
    """
    return np.random.randint(rangest,rangeend,size=n)

# def fitness_of_element()