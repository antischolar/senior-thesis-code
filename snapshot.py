import numpy as np

class snapshot:
    def __init__(self):
        self.b = []
        self.wealth = []
    
    def snapshot(self, w, money):
        self.b.append(w.tolist())
        self.wealth.append(money)
        
    def __str__(self):
        return 'wealth growth: ' + str(self.wealth) + '\n' + \
        'stock allocation: ' + str(self.b)
