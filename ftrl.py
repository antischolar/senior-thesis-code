import numpy as np
from snapshot import snapshot

class ftrl:

    SAMPLES = 5e2

    def __init__(self, n, l):
        self.b = np.array([1/n] * n)
        self.l = l
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

    def step(self, x):
        self.b = self.find_regularized_leader()
        self.wealth = self.wealth * self.recent_performance(x)
        self.take_snapshot()
        self.stock_values.append(x)
        self.loss += self.observe_loss(self.b, np.array(x))
        
    def find_regularized_leader(self):
        if len(self.stock_values) == 0:
            return self.b
        samples = self.sample_from_probability_simplex()
        min_portfolio = samples[0]
        min_loss = np.Infinity
        for sample in samples:
            w = sample
            total_performance = 0
            for stock_performance in self.stock_values:
                total_performance += self.observe_loss(w, np.array(stock_performance))
            total_performance += self.l * self.regularizer_function(w)
            if total_performance < min_loss:
                min_portfolio = w
                min_loss = total_performance
        return min_portfolio


    def observe_loss(self, w, x):
        return -np.log(np.dot(w, x))
    
    def recent_performance(self, x):
        return np.dot(self.b, x)

    def regularizer_function(self, w):
        running_sum = 0
        for i in w:
            running_sum += -np.log(i)
        return running_sum


    # draws SAMPLES number of random probability simplexes
    def sample_from_probability_simplex(self):
        n = len(self.b)
        samples = np.empty((int(self.SAMPLES), n))

        for i in range(int(self.SAMPLES)):
            sample = np.random.rand(n)
            samples[i] = np.divide(sample, sum(sample))
        return samples

