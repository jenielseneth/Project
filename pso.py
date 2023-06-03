import numpy as np
import math

def vel_update(w, a_ind, a_grp, v, pbest, pbestneighbor, p, random_grp):
    random_ind = np.random.rand()
    v_new = w*v+a_ind*random_ind*(pbest-p)+a_grp*random_grp*(pbestneighbor-p)
    # print("this is the pbesttneighbor difference: ", np.mean((pbestneighbor-p)**2))
    return v_new

class Individual:
    def __init__(self, p, neighbors, a_ind):
        self.p = p
        self.neighbors = neighbors
        self.p_nb_best = np.zeros(self.p.shape).flatten()
        self.a_ind = a_ind
        self.p_best = p
        self.v = np.zeros(p.shape)

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors
        return

    def get_neighbors(self):
        return self.neighbors
    
    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        return
    
    def set_p(self, p):
        self.p = p
        return

    def eval_self(self, p, opt_func):
        if opt_func(p) < opt_func(self.p_best):
            self.p_best = p
        return
    
    def get_current_nb_best(self, opt_func):
        best_value = opt_func(self.p)
        for neighbor in self.neighbors:
            neighbor_value = opt_func(neighbor.p)
            # print("p_best_neighbors",best_value, "vs:", neighbor_value)
            if(math.isnan(best_value) or (neighbor_value < best_value)):
                # print("previous best is: \n", p_best_neighbors)
                best_value = neighbor_value
                # print("new best is: \n", p_best_neighbors)
                # print("new best value is: ", best_value)
                # print("changing best neighbor value!: ", best_value)
    
        print("Best value for this epoch is ", best_value)
        return
    
    def get_nb_best(self, opt_func):
        best_value = opt_func(self.p_nb_best)
        for neighbor in self.neighbors:
            neighbor_value = opt_func(neighbor.p_best)
            # print("p_best_neighbors",best_value, "vs:", neighbor_value)
            if(math.isnan(best_value) or (neighbor_value < best_value)):
                # print("previous best is: \n", p_best_neighbors)
                self.p_nb_best = neighbor.p_best
                best_value = neighbor_value
                # print("new best is: \n", p_best_neighbors)
                # print("new best value is: ", best_value)
                # print("changing best neighbor value!: ", best_value)
    
        print("Best neighbour value is ", best_value)
        return self.p_nb_best

    def set_nb_best(self, nb_best):
        self.p_nb_best = nb_best
        return
    
    def v_update(self, w, a_grp, random_grp):
        self.v = vel_update(w, self.a_ind, a_grp, self.v, self.p_best, self.p_nb_best, self.p, random_grp)
        return
    
    def p_update(self):
        p = self.p + self.v
        self.p = p
        return 
    
def optimize(opt_func, p_up_bound, p_low_bound, initial_value, num_ind = 10, num_neighbors=9, epochs = 10):
    assert num_neighbors < num_ind
    w = 1
    a_grp = 0.15 # np.random.rand()
    a_inds = np.random.rand((num_ind)) #np.ones((num_ind))
    swarm = []
    for i in range(0, num_ind):
        p = np.random.default_rng().uniform(p_low_bound, p_up_bound, len(p_low_bound))+initial_value
        swarm.append(Individual(p, [], a_inds[i]))
    
    print("Setting up neighbors")
    #set neighbors
    neighbor_count = np.full((num_ind, num_neighbors), -1)
    for ind in swarm:
        index = swarm.index(ind)
        ind_neighbors = []
        all_neighbors = np.arange(num_ind)
        all_neighbors = np.delete(all_neighbors, index)
        if(num_ind==num_neighbors+1):
            neighbor_count[index] = all_neighbors
        else:
            for i in range(0, num_neighbors):
                # nb = np.random.randint(num_ind)
                # while (neighbor_count[index, i] > num_neighbors or nb in neighbor_count[index] or nb == index):
                #     nb = np.random.randint(num_ind)
                nb = np.random.choice(all_neighbors)
                index_of_nb = np.where(all_neighbors == nb)
                all_neighbors = np.delete(all_neighbors, index_of_nb)

                neighbor_count[index, i] = nb
                # neighbor_count[nb, i] = index
    
    i = 0
    for ind in neighbor_count:
        for neighbor in ind:
            swarm[i].add_neighbor(swarm[neighbor])
        i+=1
    
    p_nb_prev_best = np.zeros((len(p_low_bound)))
    up_threshold = 0.95
    down_threshold = 0.5
    for epoch in range(epochs):
        print("Epoch:", epoch+1)
        for ind in swarm:
            ind.eval_self(np.array(ind.p).flatten(), opt_func)
        
        p_nb_best = swarm[0].get_nb_best(opt_func)

        for ind in swarm:
            if (num_neighbors == num_ind-1):
                ind.set_nb_best(p_nb_best)
            else :
                ind.get_nb_best(opt_func)


        # if(opt_func(p_nb_best)/opt_func(p_nb_prev_best) > up_threshold):
        #     w *= 1.05
        # elif(opt_func(p_nb_best)/opt_func(p_nb_prev_best) < down_threshold):
        #     w*=0.8
        # else:
        #     w = 1
        # print("w is: ", w)
        for ind in swarm:
            random_grp = np.random.rand()
            # if(opt_func(p_nb_best)/opt_func(p_nb_prev_best) > up_threshold):
            #     random_grp*=w
            ind.v_update(w, a_grp, random_grp)

        for ind in swarm:
            # print("old value is: ", opt_func(ind.p))
            ind.p_update()
        swarm[0].get_current_nb_best(opt_func)
        p_nb_prev_best=p_nb_best
    
    #get best
    p_best = np.zeros(len(swarm[0].p)).flatten()
    for ind in swarm:
        if (opt_func(ind.p_best) < opt_func(p_best)):
            p_best = ind.p_best
    return p_best


