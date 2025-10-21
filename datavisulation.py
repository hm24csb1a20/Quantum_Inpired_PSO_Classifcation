import pandas as pd
import matplotlib.pyplot as plt
import os
from pandas.plotting import scatter_matrix
import math



current = os.getcwd()
file_path = os.path.join(current, "data", "student-mat.csv")

df = pd.read_csv(file_path, sep=';')  
# print(df.head())
last_col = df.columns[-1]
other_cols = df.columns[:-1]
# make the classifcation col
df['above_12'] = df[last_col] > 11

# Split into batches
cols_per_fig = 3
num_figs = math.ceil(len(other_cols) / cols_per_fig)

for fig_idx in range(num_figs):
    start = fig_idx * cols_per_fig
    end = start + cols_per_fig
    batch_cols = other_cols[start:end]
    
    fig, axes = plt.subplots(nrows=len(batch_cols), ncols=1, figsize=(6, 3*len(batch_cols)))
    
    if len(batch_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(batch_cols):
        # Use c=boolean array with a colormap
        scatter = axes[i].scatter(df[col], df[last_col], c=df['above_12'], cmap='bwr')  # blue-white-red
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(last_col)
        axes[i].set_title(f'{col} vs {last_col}')
    
    # Add a colorbar once per figure
    fig.colorbar(scatter, ax=axes, orientation='vertical', label='Above 12')
    
    plt.tight_layout()
    plt.show()

# the observations noted are present in the readme file 