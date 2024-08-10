import numpy as np

# define objective function
def f1(x):
    return np.power(x, 2)[0]


# define objective function
def f2(x):
    return x[0]**2.0 + x[1]**2.0


# define callback
class CountEpochs:
    
    def __init__(self) -> None:
        self.epoch = 0
        
    def callback(self, epoch:int, fitness:list, population:list) -> bool:
        self.epoch += 1
        return False


# define sphere function
def sphere(x):
    return np.sum(np.power(x, 2))