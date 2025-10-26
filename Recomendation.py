import numpy as np



class Recomendation:


    def __init__(self, ratings, test_ratings= None):

        self.ratings = ratings
        self.num_users = ratings.shape[0]
        self.num_items = ratings.shape[1]

        # Si no se pasa test raintgs la creamos vacía
        if test_ratings is None:
            self.test_ratings = test_ratings


    def get_items(self, u, test= False):
        '''Devuelve los items que ha votado un usuario.'''
        if test:
            return np.where(~np.isnan(self.test_ratings[u]))[0]
        else:
            return np.where(~np.isnan(self.ratings[u]))[0]
    

    def get_common_items(self, u, v):
        '''Devuelve los items que tienen en común dos usuarios "u" y "v".'''
        return np.where((~np.isnan(self.ratings[u])) & ~np.isnan(self.ratings[v]))[0]