# Quantum_Inpired_PSO_Classifcation

## Phase 1:
*Classify if the students G3 tests are greater than 10 (pass/fail)*
<details> 
<summary>
See the dataset attributes 
</summary> 
<img src="./data/datasetattributes.png" width="500
">
</details>

### Observations made of the data
Based on the graphs in the `datavisulation.py` file the following observations were made
1. failures > 3 `->` fail
2. Higher==no `->` fail
3. Dalc ==  4 || 5 `->` fail
4. absences >25 `->` fail
5. G1 >= 14 `->` pass
6. G2 >=14 `->` pass

---
### The QIGPSO Algo for Feature Selection 

#### 1.  Population intialization
this makes the population of [0/1] vector of the size of the no of featurs/col-length of the data

#### 2.  Fitness Function
This determines the Exploitive/Explorative nature of the algo 
<br>
For each element the fitness is calcuted as:
<br>
$$
\text{f} = \alpha \log(\text{acc}) - (1 - \alpha) \frac{\text{n\_feat}}{\sqrt{N}} (1 + \sin(\phi))
$$

#### 3. Determine Best/Worst Fitness and Mass
The fitness values are used to calculate the **inertial mass ($\text{Mi}$)** and **normalized mass ($\text{mi}$)** for each particle.

