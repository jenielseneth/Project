import sys
from pandas import Timestamp
import yfinance as yf
# Import the plotting library
import matplotlib.pyplot as plt
import numpy as np
import datetime
from scipy.optimize import fmin
from scipy.optimize import minimize

import pso
import linearsquares
import ga
np.set_printoptions(suppress=True)


def lppl(x, a, b, t, m, c, w, phi):
    return (a-b*np.power((t-x),m)*(1+c*np.cos(w*np.log(t-x)+phi))) #(a+b*np.power((t-x),m)+c*np.power((t-x),m)*np.cos(w*np.log(t-x)-phi))
vlppl = np.vectorize(lppl)

def opt_params(x):
    A=x[0]
    B=x[1]
    T=np.round(x[2]).astype(int)
    m=x[3]
    C=x[4]
    w=x[5]
    phi=x[6]
    loss = 0
    if(T == 0) : return sys.maxsize
    # print("in opt_params x is ", x)
    for i in range(len(data_ln)):
        #using the function from the LPPL video
        loss += (data_ln[i]-(A-B*np.power((T-i),m)*(1+C*np.cos(w*np.log(T-i)+phi))))**2#loss += np.linalg.norm(data_ln[i]-(A+B*np.power((T-i),m)+C*np.power((T-i),m)*np.cos(w*np.log(T-i)-phi)))
    return loss



# Get the data for the S&P 500
data_jump = 50
start_date = Timestamp('2003-07-01')
start_date = datetime.date(2003, 7, 1)
complete_data = yf.download('^GSPC',start_date,'2007-07-01')
adj_close_data = complete_data['Adj Close']
data = adj_close_data.values
data_ln = np.log(data)


#get i, j, k
i_date=Timestamp('2006-05-05')
j_date=Timestamp('2007-02-20')
k_date=Timestamp('2007-06-04')
i= complete_data.index.get_loc("2006-05-05")+1
j= complete_data.index.get_loc("2007-02-20")+1
k= complete_data.index.get_loc("2007-06-04")+1
print("i: ", i, "j: ", j, "k: ", k)

#calculations
rho = ((j-i))/((k-j))
T = (rho*k-j)/(rho-1)
omega = 2*np.pi/np.log(rho)
phi = np.pi - omega*np.log(T-k)
print("Rho: ", rho, "T: ", T, "Omega: ", omega, "Phi: ", phi)


print("linear", linearsquares.get_A_B_C(data_ln, T, omega, phi, 0.5))


#get exponential fit:
b_column = np.arange(len(complete_data))+1
b_column = -(np.round(T).astype(int)-b_column)
X = np.vstack([np.ones(len(complete_data)), b_column]).T
A, B = np.linalg.lstsq(X, data_ln, rcond=None)[0]

exp_params = [A, B, T, 1, 0, 0, 0]

#Optimal params
optimized_params = [7.47, 0.014, 1105.8, 0.52, -0.04, 9.87, 2.72]

#Downhill Simplex algorithm
#x is initial starting point
x = [A, B, T, 0.5, 0, omega, phi]
downhill_params = minimize(opt_params, x, method='Nelder-Mead').x


def ga_cost_func(stacked_params):
    costs = []
    for params in stacked_params:
        A, B, C = linearsquares.get_A_B_C(data_ln, params[0], params[1], params[2], params[3])
        costs.append(opt_params([A, B, params[0], params[3], C, params[1], params[2]]))
    return np.array(costs)
        

#GA algorithm
print("Calculating with the GA algorithm...")
ga_lower_bound=np.array([len(data_ln),  6, 0, 0.1])
ga_upper_bound=np.array([T+300,  15, 2*np.pi, 0.9])
ga_total_params = ga.ga(ga_lower_bound, ga_upper_bound, ga_cost_func, epochs=10, num_pop=50, num_children=25, num_mutations=25)
# best_params = np.zeros((7))
# best_loss = opt_params(best_params)
# for params in ga_total_params[0:10]:
#     print("downhill")
#     A, B, C = linearsquares.get_A_B_C(data_ln, params[0], params[1], params[2], params[3])
#     ga_params = [A, B, params[0], params[3], C, params[1], params[2]]
#     ga_param_downhill = minimize(opt_params, ga_params, method='Nelder-Mead').x
#     loss = opt_params(ga_param_downhill)
#     if (loss < best_loss):
#         best_params = ga_param_downhill
#         best_loss = loss
# ga_params = best_params
ga_params = ga_total_params[0]
A, B, C = linearsquares.get_A_B_C(data_ln, ga_params[0], ga_params[1], ga_params[2], ga_params[3])
ga_params = [A, B, ga_params[0], ga_params[3], C, ga_params[1], ga_params[2]]


# PSO algorithm
print("Calculating with the PSO algorithm...")
bound = 4
initial_lower_bound = np.array([ -bound, -bound, -bound, -bound, -bound, -bound, -bound])
initial_upper_bound = initial_lower_bound*(-1)
lower_bound=np.array([-bound, -x[1], x[2]-len(data_ln), -0.5, -1, -omega/4, -phi])
upper_bound=np.array([bound,   bound, bound*25, 0.5, 0.5,  omega/4,  2*np.pi-phi])
pso_params = pso.optimize(opt_params, upper_bound, lower_bound, initial_value=x, num_ind=300,num_neighbors=299, epochs=30)


#GA-PSO
print("Calculating with the GA-PSO algorithm...")
lower_bound=np.array([-1, -ga_params[1], ga_params[2]-len(data_ln), -0.3, -1, -omega/4, -phi])
upper_bound=np.array([1,   0.05, bound*25, 0.3, 1,  omega/4,  2*np.pi-phi])
ga_pso_params = pso.optimize(opt_params, upper_bound, lower_bound, initial_value=ga_params, num_ind=300,num_neighbors=299, epochs=10)

#PSO-GA
print("Calculating with the PSO-GA algorithm...")
ga_lower_bound=np.array([len(data_ln),  pso_params[5]-1, 0, pso_params[3]-0.3])
ga_upper_bound=np.array([pso_params[2]+20,  pso_params[5]+1, 2*np.pi, pso_params[3]+0.3])
ga_total_params = ga.ga(ga_lower_bound, ga_upper_bound, ga_cost_func, epochs=10, num_pop=100, num_children=50, num_mutations=50)
pso_ga_params = ga_total_params[0]
A, B, C = linearsquares.get_A_B_C(data_ln, pso_ga_params[0], pso_ga_params[1], pso_ga_params[2], pso_ga_params[3])
pso_ga_params = [A, B, pso_ga_params[0], pso_ga_params[3], C, pso_ga_params[1], pso_ga_params[2]]

#PSO -> Downhill
# pso_down_params = minimize(opt_params, pso_params, method='Nelder-Mead').x




#losses
exp_loss = opt_params(exp_params)
downhill_loss = opt_params(downhill_params)  
optimal_loss = opt_params(optimized_params)
ga_loss = opt_params(ga_params)
pso_loss = opt_params(pso_params)
ga_pso_loss = opt_params(ga_pso_params)
pso_ga_loss = opt_params(pso_ga_params)
# pso_down_loss = opt_params(pso_down_params)

#all four combined
raw_losses = np.array([1/ga_loss, 1/pso_loss, 1/ga_pso_loss, 1/pso_ga_loss])
normalized_losses = np.array([float(i)/sum(raw_losses) for i in raw_losses])
print("this is normalized losses: ", normalized_losses)
all_params = np.array([ga_params, pso_params, ga_pso_params, pso_ga_params])
comb_params = np.dot(normalized_losses, all_params)
comb_loss = opt_params(comb_params)


#print statements
print("Exponential loss is: ",exp_loss ,"Downhill loss is: ", downhill_loss, "PSO loss is: ", pso_loss, "Optimal Loss is: ", optimal_loss, "PSO-Downhill loss is: ", "pso_down_loss", "GA loss is: ", ga_loss, "GA-PSO loss is: ", ga_pso_loss, "Comb loss is: ", comb_loss)
print("Params are: [A , B , T , m , C , omega , phi]")
print("omega: osc, ")
print("The optimal paramters are", optimized_params)
print("The optimized paramters with Downhill Simplex are: \n", np.round(downhill_params, 5))
print("The optimized paramters with GA are: \n", np.round(ga_params, 5))
print("The optimized paramters with PSO are: \n", np.round(pso_params, 5))
print("The optimized paramters with GA-PSO are: \n", np.round(ga_pso_params, 5))
print("The optimized paramters with PSO-GA are: \n", np.round(pso_ga_params, 5))
print("The optimized paramters with combining are: \n", np.round(comb_params, 5))
# print("The optimized paramters with PSO-Down are: \n", np.round(pso_down_params, 5))

def print_critical_time(a, name):
    print("The critical time for ", name, " is: ", a[2])

print("The length of the dataset is: ", len(data_ln))
print_critical_time(downhill_params, "Downhill")
print_critical_time(optimized_params, "Optimal")
print_critical_time(ga_params, "GA")
print_critical_time(pso_params, "PSO")
print_critical_time(ga_pso_params, "GA-PSO")
print_critical_time(pso_ga_params, "PSO-GA")
print_critical_time(comb_params, "Combine")
# print_critical_time(pso_down_params, "PSO-Down")

#get log function
log_optimal_func= vlppl(range(data_ln.size), optimized_params[0], optimized_params[1], optimized_params[2], optimized_params[3], optimized_params[4], optimized_params[5], optimized_params[6])
log_ga_func = vlppl(range(data_ln.size), ga_params[0], ga_params[1], ga_params[2], ga_params[3], ga_params[4], ga_params[5], ga_params[6])
log_downhill_func = vlppl(range(data_ln.size), downhill_params[0], downhill_params[1], downhill_params[2], downhill_params[3], downhill_params[4], downhill_params[5], downhill_params[6])
# log_pso_down_func = vlppl(range(data_ln.size), pso_down_params[0], pso_down_params[1], pso_down_params[2], pso_down_params[3], pso_down_params[4], pso_down_params[5], pso_down_params[6])
log_pso_func = vlppl(range(data_ln.size), pso_params[0], pso_params[1], pso_params[2], pso_params[3], pso_params[4], pso_params[5], pso_params[6])
log_ga_pso_func = vlppl(range(data_ln.size), ga_pso_params[0], ga_pso_params[1], ga_pso_params[2], pso_params[3], ga_pso_params[4], ga_pso_params[5], ga_pso_params[6])
log_pso_ga_func = vlppl(range(data_ln.size), pso_ga_params[0], pso_ga_params[1], pso_ga_params[2], pso_ga_params[3], pso_ga_params[4], pso_ga_params[5], pso_ga_params[6])
log_comb_func = vlppl(range(data_ln.size), comb_params[0], comb_params[1], comb_params[2], comb_params[3], comb_params[4], comb_params[5], comb_params[6])


adj_close_data.plot()
plt.plot(complete_data.index, np.exp(log_optimal_func), color='green')
plt.plot(complete_data.index, np.exp(log_downhill_func), color='yellow')
plt.plot(complete_data.index, np.exp(log_ga_func), color='red')
plt.plot(complete_data.index, np.exp(log_pso_func), color='blue')
plt.plot(complete_data.index, np.exp(log_ga_pso_func), color='magenta')
plt.plot(complete_data.index, np.exp(log_pso_ga_func), color='cyan')
# plt.plot(complete_data.index, np.exp(log_comb_func), 'g--')
# plt.plot(complete_data.index, np.exp(log_pso_down_func), color='red')
plt.legend(['Time Series', 'Target Function', 'Downhill Simplex', 'GA', 'PSO', 'GA-PSO', 'PSO-GA', 'Combined'])
plt.plot(i_date, adj_close_data.loc[i_date], 'o', color='black')
plt.plot(j_date, adj_close_data.loc[j_date], 'o', color='black')
plt.plot(k_date, adj_close_data.loc[k_date], 'o', color='black')
plt.show()