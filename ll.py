import numpy as np
from snapshot import snapshot

class ll:
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

    def step(self, x):
        self.b = self.find_max()
        self.wealth = max(0, self.wealth * self.recent_performance(x))
        self.take_snapshot()

        self.stock_values.append(x)
        self.loss += self.observe_loss(self.b, np.array(x))

    def find_max(self):
        if len(self.stock_values) == 0:
            return self.b
        n = len(self.b)
        portfolio = np.zeros(n)
        max_ind = 0
        max_growth = 0

        if len(self.stock_values) != 0:
            for i in range(n):
                if self.stock_values[-1][i] > max_growth:
                    max_growth = self.stock_values[-1][i]
                    max_ind = i

        portfolio[max_ind] = 1
        return portfolio

    def observe_loss(self, w, x):
        return -np.log(np.dot(w, x))

    def recent_performance(self, x):
        return np.dot(self.b, x)