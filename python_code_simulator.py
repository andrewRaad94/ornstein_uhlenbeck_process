
from math import exp,sqrt
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pandas as pd


class simulator:
    def __init__(self) -> None:
        pass 

    def run_simulations(self, n : int ):
        '''
        Determine how many Ornstein-Ohlenbeck simulations you would like to generate
        
        Returns
        --------
        'dict' 
            A dictionary in which the key, value pairs are the index of each simulation and a list of the generated outcomes
        
        '''
        results  = {}
        for s in range(self.simulations):
            results[s] = self.calculate()
        return results

    def plot_simulations(self, results):
        '''
        Plot simulations once they are ready
        '''
        results = pd.DataFrame(results)
        plt.figure(figsize=(15, 10))
        plt.plot(results)
        self.add_graph_information()
        plt.show()
        return
    
    def __call__(self):
        '''
        Call method to generate and plot all simulations after all variables are created
        '''
        simulations = self.run_simulations(self.simulations)
        self.plot_simulations(simulations)
        return
    
    def k_sensitivity(self, k_list):
        results = {}
        self.k_list = k_list
        for k in k_list:
            self.k = k
            results['k ='+str(k)] = self.calculate()
        
        pdb.set_trace()
        plt.figure(figsize=(15, 10))
        results = pd.DataFrame(results)
        plt.plot(results, label = results.columns)
        plt.legend()
        plt.title(f"K sensitivity {self.process_name} using $\mu = {self.mean}$ and $\sigma = {self.sigma}$")
        plt.show()




class ornstein_uhlenbeck_simulation(simulator):
    '''
    This class once instantiated will generate and plot price paths that were simulated with Euler-Maruyana Method.
        Parameters
    ----------
    initial value : `float`
        The initial value of the process at time 0.
    mean : `float`
        Expected value of the process.
    sigma : `float`
        Standard deviation.
    k : `float`
        Velocity of reversion to mean.
    increments : `float`
        how many intervals during the time horizon.
    '''

    def __init__(self, initial_value, mean, sigma, k, increments,time_interval, n_simulations):
        super().__init__()
        self.initial_value = initial_value
        self.mean = mean 
        self.sigma = sigma 
        self.k = k
        self.increments = increments
        self.time_interval = time_interval
        self.simulations = n_simulations
        self.process_name = 'Ornstein Uhlenbeck Process'

    def calculate(self):
        '''
        Generate the path of the stock using gbm
        '''
        increments_per_period = int(self.increments/ self.time_interval)
        dt = 1/increments_per_period #for each period, there will be dt time steps
        value_array = np.zeros(self.increments+1)
        value_array[0] = self.initial_value

         
        for i in range(1, self.increments+1):
            previous_value = value_array[i-1]
            value_array[i] = previous_value + self.k*(self.mean - previous_value) * dt + self.sigma * np.random.normal(0, np.sqrt(dt))
        return value_array 

    def add_graph_information(self):
        plt.xlabel("Increments")
        plt.ylabel("Stock Value")
        plt.title(f"Realizations of {self.simulations} Ornstein Uhlenbeck Process simulations using $\mu = {self.mean}$,$k = {self.k}$ and $\sigma = {self.sigma}$")




class geometric_mean_reversion_simulation(simulator):
    '''
    This class once instantiated will generate and plot price paths that were simulated with Euler-Maruyana Method.
        Parameters
    ----------
    initial value : `float`
        The initial value of the process at time 0.
    mean : `float`
        Expected value of the process.
    sigma : `float`
        Standard deviation.
    k : `float`
        Velocity of reversion to mean.
    increments : `float`
        how many intervals during the time horizon.
    '''

    def __init__(self, initial_value, mean, sigma, k, increments,time_interval, n_simulations):
        super().__init__()
        self.initial_value = initial_value
        self.mean = mean 
        self.sigma = sigma 
        self.k = k
        self.increments = increments
        self.time_interval = time_interval
        self.simulations = n_simulations
        self.process_name = 'Geometric Mean Reversion Process'

    def calculate(self):
        '''
        Generate the path using geometric mean reversion
        '''
        increments_per_period = int(self.increments/ self.time_interval)
        dt = 1/increments_per_period #for each period, there will be dt time steps
        value_array = np.zeros(self.increments+1)
        value_array[0] = self.initial_value

         
        for i in range(1, self.increments+1):
            previous_value = value_array[i-1]
            # value_array[i] = previous_value *(1+ (self.k*(self.mean - np.log(previous_value))) * dt + self.sigma * np.random.normal(0, np.sqrt(dt)))
            value_array[i] = previous_value+ self.k*(self.mean - np.log(previous_value))* previous_value*dt + self.sigma* previous_value * np.random.normal(0, np.sqrt(dt))
        return value_array

    def add_graph_information(self):
        plt.xlabel("Increments")
        plt.ylabel("Y Values")
        plt.title(f"Realizations of {self.simulations} Geometric Mean Reversion Process simulations using $\mu = {self.mean}$, $k = {self.k}$ and $\sigma = {self.sigma}$")


if __name__ == "__main__":
    gm = geometric_mean_reversion_simulation(1, 1.5, 0.4, 15, 252,1, 5)
    # gm()
    k_list = [1, 2,5, 15]
    gm.k_sensitivity(k_list)
    # ou.k_sensitivity(k_list)