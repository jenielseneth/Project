import numpy as np

def initialize_params(upper_bound, lower_bound, num):
    init_params = np.random.default_rng().uniform(lower_bound, upper_bound, (num,len(lower_bound)))
    return init_params

def breeding_mech(num_children, parents):
    rng = np.random.default_rng()
    children = []
    for i in range(num_children):
        p1,p2 = rng.choice(parents, 2, replace=False)
        child = np.sqrt((p1+p2)**2/2)
        children.append(child)
    return np.array(children)

def mutation_mech(upper_bound, lower_bound, population, num_mut, k = 2):
    stat_range = upper_bound-lower_bound
    dist_upper = k*stat_range
    dist_lower = -k*stat_range
    mutated_pop = []
    rng = np.random.default_rng()
    for i in range(num_mut):
        picked = rng.choice(population, 1, replace=False)
        perturb_vector = np.random.default_rng().uniform(dist_lower, dist_upper, len(dist_lower))
        mutated_pop.append((picked+perturb_vector).flatten())
    return np.array(mutated_pop)

def culling_mech(old_pop, children, mutations, cost_func):
        new_pop = np.concatenate(( children, mutations, old_pop))
        costs = cost_func(new_pop)
        sorted_indices = np.argsort(costs)
        top_indices = sorted_indices[0:round(old_pop.shape[0])]
        return new_pop[top_indices]

def ga(lower_bound, upper_bound, cost_func, epochs=10, num_pop=50,  num_mutations = 25, num_children = 25, k=2):
    population = initialize_params(upper_bound, lower_bound, num_pop)
    for epoch in range(epochs):
        children = breeding_mech(num_children, population)
        mutated = mutation_mech(upper_bound, lower_bound, population, num_mutations, k)
        population = culling_mech(population, children, mutated, cost_func)
        print("With epoch ", epoch, "the current best is: ", cost_func([population[0]]))
    return population
     



