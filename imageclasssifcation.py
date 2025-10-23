import tensorflow as tf
import numpy as np
import os
current = os.getcwd()

# read all the files in a folder
for files in os.listdir(os.path.join(current,"data",""))