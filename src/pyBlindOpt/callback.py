# coding: utf-8


'''
Ready to use callbacks.
'''


__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


class EarlyStopping:
    
    def __init__(self, threshold:float=0.0) -> None:
        self.epoch = 0
        self.threshold = threshold
    
    def callback(self, epoch:int, fitness:list, population:list) -> bool:
        self.epoch = epoch
        
        best_fitness = min(fitness)
        
        if best_fitness < self.threshold:
            return True
        else:
            return False


class CountEpochs:
    
    def __init__(self) -> None:
        self.epoch = 0
        
    def callback(self, epoch:int, fitness:list, population:list) -> bool:
        self.epoch += 1
        return False