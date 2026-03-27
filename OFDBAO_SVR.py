import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score

def evaluate_svr(params, X_train, y_train, X_test, y_test, verbose=False):
    C = abs(params[0])
    gamma = abs(params[1])
    epsilon = abs(params[2])
    try:
        # Define model
        model = svm.SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        # Time-series aware CV (no shuffling)
        cv = TimeSeriesSplit(n_splits=3)
        # Cross-validation MSE (negative values returned by sklearn)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        # Normalize RMSE to [0,1] range for consistent fitness scaling
        # based on range of target variable
        y_range = np.max(y_train) - np.min(y_train)
        if y_range > 0:
            norm_rmse = rmse / y_range
        else:
            norm_rmse = rmse  # fallback in case of constant data
        # Optional progress printing
        if verbose:
            print(f"C={C:.3f}, gamma={gamma:.6f}, epsilon={epsilon:.4f}, RMSE={rmse:.6f}, NormRMSE={norm_rmse:.6f}")
        return norm_rmse  # lower fitness = better model
    except Exception as e:
        if verbose:
            print(f"Error evaluating SVR: {e}")
        return float("inf")  # Penalize invalid parameter sets

def SpaceBound(X, Up, Low):
    X = np.where(X > Up, np.random.rand(*X.shape) * (np.array(Up) - np.array(Low)) + np.array(Low), X)
    X = np.where(X < Low, np.random.rand(*X.shape) * (np.array(Up) - np.array(Low)) + np.array(Low), X)
    return X

def get_params(maxIteration, iter):
    if iter > maxIteration:
        w = 0.5
    else:
        fx = round(maxIteration / 1)  # f is 1 in the MATLAB code
        y = iter % fx
        w = (y / fx * -0.6) + 0.6
    return w

def obl(x, xmin, xmax):
    return xmin + xmax - x;

def dFDB(population, fitness, maxIter, iter):
    # Find the index of the individual with the minimum fitness
    bestIndex = np.argmin(fitness)
    #best will store the position of the best agent in each dimension
    best = population[bestIndex]
    populationSize, dimension = population.shape
    # Initialize arrays
    distances = np.zeros(populationSize)
    normFitness = np.zeros(populationSize)
    normDistances = np.zeros(populationSize)
    divDistances = np.zeros(populationSize)
    if np.min(fitness) == np.max(fitness):
        # If all fitness values are the same, randomly select an individual
        index = np.random.randint(populationSize)
    else:
        # Get the weighting parameter
        w = get_params(maxIter, iter)
        # Calculate distances
        for i in range(populationSize):
            distances[i] = np.sum(np.abs(best - population[i]))
        minFitness = np.min(fitness)
        maxMinFitness = np.max(fitness) - minFitness
        minDistance = np.min(distances)
        maxMinDistance = np.max(distances) - minDistance
        # Normalize fitness and distances
        normFitness = 1 - (fitness - minFitness) / maxMinFitness
        normDistances = (distances - minDistance) / maxMinDistance
        # Calculate divergence distances
        divDistances = (1 - w) * normFitness + w * normDistances
        # Find the index of the maximum divergence distance
        index = np.argmax(divDistances)
    return index

def smart_restart(population, fitness, xmin, xmax):
    pop_size, dim = population.shape # N, Dim
    restart_ratio = 0.7 # 0.3 means 30 percent of population will be restart
    num_to_restart = int(restart_ratio * pop_size)
    if num_to_restart <= 0:
        return population
    worst_indices = np.argsort(fitness)[-num_to_restart:] # the agents whose fitness is less
    xmin_arr = np.array(xmin)
    xmax_arr = np.array(xmax)
    
    new_solutions = obl(population[worst_indices], xmin_arr, xmax_arr)
    population[worst_indices] = np.clip(new_solutions, xmin_arr, xmax_arr)
    return population

# OFDBAO Algorithm
def OFDBAO(X_train, X_test, y_train, y_test, N, M_Iter, LB, UB, Dim, ii):
    print("OFDBAO Working")
    Conv_curve = np.zeros(M_Iter)
    if not isinstance(LB, list):
        LB = [LB] * Dim
    if not isinstance(UB, list):
        UB = [UB] * Dim
    # make bounds arrays for safe arithmetic
    LB_arr = np.array(LB, dtype=float)
    UB_arr = np.array(UB, dtype=float)
    # Initialization
    Population = np.zeros((N, Dim))
    for i in range(Dim):
        Population[:, i] = (np.random.uniform(0, 1, N) * (UB[i] - LB[i]) + LB[i])
    fitness = np.array([evaluate_svr(ind, X_train, y_train, X_test, y_test) for ind in Population])
    iterations = []
    accuracy = []
    fitness_values = []
    position_values = []
    # Keep track of best
    best_idx = int(np.argmin(fitness))
    Best_P = Population[best_idx].copy()
    Best_FF = float(fitness[best_idx])
    Ffun_new = np.zeros(Population.shape[0]) # Initialize the new fitness values
    # Parameters for the AOA algorithm
    Alpha = 5
    #Mu = 0.499
    MOP_Max = 1
    MOP_Min = 0.2
    C_Iter = 1
    stagnation_count = 0
    sl = 30
    epsilon = 1e-2 # thershold for improvement detection
    # Main optimization loop
    while C_Iter <= M_Iter:
        #stagnation_limit2 will decrease from 30 to 0, 
        stagnation_limit = sl - C_Iter * (sl / M_Iter)#30%
        # Calculate MOP and MOA
        MOP = 1 - ((C_Iter ** (1 / Alpha)) / (M_Iter ** (1 / Alpha))) # propability ratio 0.6 - 0
        MOA = MOP_Min + C_Iter * ((MOP_Max - MOP_Min) / M_Iter) # accelerated function 0.2 - 1
        prev_best = Best_FF
        # Adaptive perturbation (faster decay)
        # sigma is 's' in paper
        sigma = 2 * ((1 - C_Iter / M_Iter) ** 2)
        # Update positions
        for i in range(N): # For each solution
            fdbIndex = dFDB(Population, fitness, M_Iter, C_Iter)
            # step is 'a' in paper
            step = sigma * (2 * np.random.rand(Dim) - 1)
            r1 = np.random.rand()
            alpha = 0.005#+ (0.05 * ((1 - C_Iter / M_Iter) ** 2))
            u = ((UB_arr-LB_arr) * abs(step) + LB_arr)
            if r1 > MOA:
                r2 = np.random.rand()
                # difference # Noise and alpha with UB and LB generates values 10 to 0 percent
                diff = abs(Population[i, :] - Population[fdbIndex, :]) # u#*0.5
                if r2 > 0.5:
                    Population[i, :] = (diff) / ((MOP + 1e-10) * u)
                else:
                    Population[i, :] = (diff) * (MOP * u)
            else:
                r3 = np.random.rand()
                if r3 > 0.5:
                    Population[i, :] = (Best_P - MOP * u)
                else:
                    Population[i, :] = (Best_P + MOP * u )
                # Ensure solutions remain within bounds
            for j in range(Dim):
                if Population[i, j] < LB[j]:
                    Population[i, j] = LB[j] + np.random.rand() * (UB[j] - LB[j]) * alpha
                elif Population[i, j] > UB[j]:
                    Population[i, j] = UB[j] - np.random.rand() * (UB[j] - LB[j]) * alpha
            #Population[i, :] = SpaceBound(Population[i, :], UB, LB)
            '''SVR MODEL TRAINING'''
            Ffun_new[i] = evaluate_svr(Population[i, :], X_train, y_train, X_test, y_test)
            if Ffun_new[i] < fitness[i]:  # If the new solution is better than the current 
                fitness[i] = Ffun_new[i]  # Update the fitness value
            if fitness[i] < Best_FF:  # If this solution is better than the current best
                Best_FF = float(fitness[i])  # Update the best fitness value
                Best_P = np.copy(Population[i, :])  # Update the best solution
            
        if abs(prev_best - Best_FF) < epsilon:
            stagnation_count += 1
        else:
            stagnation_count = 0
        
        if stagnation_count >= stagnation_limit and stagnation_limit > (sl * 30 / 100): # 30 percent of sl. THis is because when algorithm is in the last stages, then, it exploit the search space
            print(f"Stagnation detected at iteration {C_Iter}. Smart restart triggered")
            
            Population = smart_restart(Population, fitness, LB_arr, UB_arr)
            stagnation_count = 0
        # Update convergence curve
        Conv_curve[C_Iter - 1] = Best_FF
        iterations.append(C_Iter)
        accuracy.append(1.0 / (1.0 + Best_FF))
        fitness_values.append(Best_FF)
        print('----------------Count of iterations----------------' + str(C_Iter))
        print('C and gamma:' + str(Best_P), Best_FF)
        C_Iter += 1
    print('accuracy:', accuracy)
    dfc = pd.DataFrame(fitness_values)
    dfc.to_excel(f"{ii}_OFDBAO_Fit.xlsx", index=False, header=False)
    position_values.append(Best_P)
    dfc = pd.DataFrame(position_values)
    dfc.to_excel(f"{ii}_OFDBAO_Pos.xlsx", index=False, header=False)
    best_C=Best_P[0]
    best_gamma=Best_P[1]
    best_epsilon=Best_P[2]
    return best_C, best_gamma,best_epsilon,iterations,Conv_curve