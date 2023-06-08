import numpy as np
from pandas import Timestamp
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

def plot(a, b, t, m, c, w, phi, name):
    def lppl(x, a, b, t, m, c, w, phi):
        return (a-b*np.power((t-x),m)*(1+c*np.cos(w*np.log(t-x)+phi))) #(a+b*np.power((t-x),m)+c*np.power((t-x),m)*np.cos(w*np.log(t-x)-phi))
    vlppl = np.vectorize(lppl)

    start_date = Timestamp('2003-07-01')
    start_date = datetime.date(2003, 7, 1)
    complete_data = yf.download('^GSPC',start_date,'2007-07-01')
    adj_close_data = complete_data['Adj Close']
    data = adj_close_data.values
    data_ln = np.log(data)
    log_func= vlppl(range(data_ln.size), a, b, t, m, c, w, phi)
    adj_close_data.plot()
    plt.plot(complete_data.index, np.exp(log_func), color='red')
    plt.legend(['Time Series', name])

if __name__ == '__main__':
    params = [1,1,1100,0.5,0.5,0,0]
    plot(params[0],params[1],params[2],params[3],params[4],params[5],params[6], "Function")
    plt.show()