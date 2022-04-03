import numpy as np
from snapshot import snapshot

class ogd:

    def __init__(self, n, eta):
        self.n = n
        self.eta = eta
        self.b = np.array([1/n] * n)
        self.stock_values = []
        self.wealth = 1
        self.snapshot = snapshot()
        self.take_snapshot()

    def ingest(self, data):
        i = 0
        online_alloc = np.array([1/self.n] * self.n)
        for perf in data:
            online_alloc = self.step(perf, online_alloc, i)
            i += 1
        return self.snapshot

    def take_snapshot(self):
        self.snapshot.snapshot(self.b, self.wealth)

    def step(self, x, online_alloc, i):
        self.wealth = self.wealth * np.dot(self.b, x)
        self.take_snapshot()

        y = online_alloc - self.eta * self.online_gradient(x)
        self.proj(y)
        self.b = self.b * i/(i+1) + y/(i+1)
        self.stock_values.append(x)
        return y

    # x is performance for today
    def online_gradient(self, x):
        # only compute the ratio for current day / previous day
        rx = np.dot(x, self.b)
        grad = -x/rx
        
        return grad
    
    # y is a list of probabilities
    # pretty standard implementation of the algorithm linked
    def proj(self, y):
        us = np.sort(y)[::-1]
        cs = 1- np.cumsum(us)

        p = 0
        
        for i in range(us.shape[0]):
            if us[i] + 1/(i+1) * cs[i] > 0: 
                p = i
        
        lamb = 1/(p+1) * cs[p]
        
        for i in range(len(y)):
            y[i]=(max(0, y[i] + lamb))  