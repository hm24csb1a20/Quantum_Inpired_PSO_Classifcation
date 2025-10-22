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

Set FITNESS_MODE = 'exploit' to push the search toward high-accuracy subsets (more exploitation). Consider raising alpha to 0.85-0.95 for even stronger exploitation.

Set FITNESS_MODE = 'explore' to encourage exploration and diverse subsets (entropy bonus). Consider lowering alpha to 0.55-0.7 here.

#### 3. Determine Best/Worst Fitness and Mass
The fitness values are used to calculate the **inertial mass ($\text{Mi}$)** and **normalized mass ($\text{mi}$)** for each particle.

$$
\text{Mi}_i = \frac{\text{f}_i - \text{f}_{\text{best}}}{\text{f}_{\text{best}} - \text{f}_{\text{worst}} + \epsilon}
\quad \text{and} \quad
\text{mi}_i = \frac{\text{Mi}_i}{\sum_{j} \text{Mi}_j + \epsilon}
$$
- The mass is calculated relative to the **global best fitness ($\text{f}_{\text{best}}$)**, meaning particles with higher fitness exert less "pull" on the population (as they are already near the target).

#### 4. Gravitational Constant (G) and Inertia Weight ($\omega$)

These control the exploration ability of the algorithm across iterations ($i$).
- **Gravitational Constant ($\mathbf{G}$):** Decays exponentially, decreasing the gravitational influence over time to favor local search (exploitation).
$$
G(i) = G_0 \exp\left(-\alpha \frac{i}{\text{max\_iter}}\right)
$$
- **Inertia Weight ($\mathbf{\omega}$):** Decays linearly from $\omega_{\text{max}}$ to $\omega_{\text{min}}$, balancing exploration vs. exploitation in the $\text{PSO}$ component.

#### 5. Position Update (Quantum-Inspired Step)

Particles update their positions based on an **acceleration ($\mathbf{\text{acc}}$)** term, which combines the influence of the particle's best position ($\text{pbest}$) and the mean best position of the population ($\text{mbest}$). The update has a quantum-inspired nature, using a probabilistic approach to generate a new binary position:

1.  **Calculate Acceleration:** $\mathbf{\text{acc}}$ is computed based on $\text{pbest}$ and $\text{mbest}$, scaled by $\text{mi}$ and $\omega$.
2.  **Continuous Position Update:** The current position ($\mathbf{x}_i$) is updated in continuous space:
    $$
    \mathbf{x}_{\text{new}} = \mathbf{x}_i + G \cdot \mathbf{\text{acc}} \cdot (\text{Random Phase})
    $$
3.  **Probabilistic Binarization (Quantization):** The continuous position is converted into a probability using the **Sigmoid function**. This probability determines if the new feature is '1' (selected) or '0' (unselected).
    $$
    P(\text{feature} = 1) = \frac{1}{1 + e^{-\mathbf{x}_{\text{new}}}}
    $$
4.  **Random Flips (Perturbation):** A small random flip probability ($\text{flip\_prob}=0.04$) is applied to introduce necessary diversity and help escape local optima.

#### 6. Update $\text{pbest}$ and $\text{gbest}$
The fitness of the $\text{new\_population}$ is evaluated.
- A particle's **personal best ($\text{pbest}$)** is updated if the new position has a better fitness.
- The **global best ($\text{gbest}$)** position is updated if any particle finds a solution better than the current $\text{gbest}$.