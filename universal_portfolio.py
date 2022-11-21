import numpy as np
from snapshot import snapshot

class universal_portfolio:

    SAMPLES = 5e2
    # initialize the object for universal portfolio, n is the number of stocks
    def __init__(self, n):
        self.b = np.array([1/n] * n)
        self.stock_values = []
        self.wealth = 1
        self.loss = 0
        self.snapshot = snapshot()
        self.take_snapshot()

    def ingest(self, data):
        for perf in data:
            self.step(perf)
        return self.snapshot
    
    def take_snapshot(self):
        self.snapshot.snapshot(self.b, self.wealth)

    # x is the growth vector corresponding to the stock growth in this time period
    def step(self, x):
        # assert we have data for all stocks
        assert len(x) == len(self.b)

        # compute how much we have in wealth using the strategy computed in
        # previous iteration
        self.wealth = self.wealth * np.dot(self.b, x)
        self.loss += self.observe_loss(self.b, np.array(x))
        self.take_snapshot()

        # update portfolio for next iteration
        self.stock_values.append(x)
        simplex = self.sample_from_probability_simplex()
        self.b = self.weighted_average(simplex)


    # draws SAMPLES number of random probability simplexes
    def sample_from_probability_simplex(self):
        n = len(self.b)
        samples = np.empty((int(self.SAMPLES), n))

        for i in range(int(self.SAMPLES)):
            sample = np.random.rand(n)
            samples[i] = np.divide(sample, sum(sample))
        return samples

    def calculate_total_loss(self):
        total_performance = 0
        for stock_performance in self.stock_values:
            total_performance += self.observe_loss(self.b, np.array(stock_performance))
        return total_performance

    def observe_loss(self, w, x):
        return -np.log(np.dot(w, x))

    # calculates weighted average of the performance of each sample, this is
    # the integration step in Universal Portfolio
    def weighted_average(self, simplex):
        total_weight = 0
        total_vectors = np.zeros(len(self.b))
        for i in range(int(self.SAMPLES)):
            curr_b = simplex[i]
            performance = self.crp_performance(curr_b)
            performance_vector = performance * curr_b
            total_vectors = np.add(total_vectors, performance_vector)
            total_weight += performance
                
        return (total_vectors / total_weight)

    # calculates performance of this CRP for all previous stock data
    def crp_performance(self, b):
        wealth = 1
        for i in range(len(self.stock_values)):
            curr_stock_performance = np.array(self.stock_values[i])
            wealth *= np.dot(b, curr_stock_performance)
        return wealth