import numpy as np
from tqdm import tqdm
from Recomendation import Recomendation



class Matrix_factorization(Recomendation):


    def __init__(self, ratings, test_ratings, num_factors, learning_rate, regularization):
        super().__init__(ratings)
        
        self.ratings = ratings
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.num_users = ratings.shape[0]
        self.num_items = ratings.shape[1]
        self.test_ratings = test_ratings

        self.p = np.array([[np.random.random() for _ in range(num_factors)] for _ in range(self.num_users)])
        self.q = np.array([[np.random.random() for _ in range(num_factors)] for _ in range(self.num_items)])


    def compute_prediction(self, u, i):
        '''Obtiene la predicción de del usuario u, con sus factores latentes,
        al item i, con sus factores.'''

        return np.dot(self.p[u], self.q[i])


    def train(self, num_iterations):
        '''Realiza el entrenamiento para ajustar el peso de las matrices p y q'''

        for it in tqdm(range(num_iterations)):

            updated_p = self.p.copy() # clone p matrix
            updated_q = self.q.copy() # clone q matrix

            error_list = []

            for u in range(self.num_users):

                for i in self.get_items(u):
                
                    error = self.ratings[u][i] - self.compute_prediction(u, i)
                    error_list.append(abs(error))

                    updated_p[u] += self.learning_rate * (error * self.q[i] - self.regularization * self.p[u])
                    updated_q[i] += self.learning_rate * (error * self.p[u] - self.regularization * self.q[i])

            #print(f'Error: {np.mean(error)}')

            self.p = updated_p
            self.q = updated_q


    def get_user_recomendations(self, u, N, items= None, predictions= None):
        '''Obtiene los N mejores items para el usuario dado'''

        if predictions is None:
            if items is None:
                items = np.arange(self.num_items)
            predictions = np.array([self.compute_prediction(u, i) for i in items])

        predictions_copy = list(predictions)
        predictions_copy.sort(reverse= True)

        resul = []
        p = 0
        while len(resul) < N and p < len(predictions_copy):
            resul += list(items[np.where(predictions == predictions_copy[p])[0]])
            p += 1

        return resul[:N]
    

    def get_recommendations(self, N= 5, users= None, test= False, pred= False):
        '''Obtiene las N mejors recomendaciones para todos los usuarios.'''

        if test:
            return {str(u) : self.get_user_recomendations(u, N, test= test, items= self.get_items(u, test= test), predictions= self.test_ratings[u][~np.isnan(self.test_ratings[u])])\
                    for u in tqdm(users)}
        if pred:
            return {str(u) : self.get_user_recomendations(u, N, items= self.get_items(u, test= pred))\
                    for u in tqdm(users)}
        return {str(u) : self.get_user_recomendations(u, N) for u in tqdm(range(self.num_users))}
    


class Matrix_factorization_bias(Matrix_factorization):


    def __init__(self, ratings, test_ratings, num_factors, learning_rate, regularization):
        super().__init__(ratings, test_ratings, num_factors, learning_rate, regularization)

        self.bu = np.array([np.random.random() for _ in range(self.num_users)])
        self.bi = np.array([np.random.random() for _ in range(self.num_items)])

        self.rating_avg = self.__get_rating_average()


    def __get_rating_average(self):
        rating_average = 0
        rating_count = 0

        for u in range(self.num_users):
            for i in self.get_items(u):
                rating_average += self.ratings[u][i]
                rating_count += 1

        rating_average /= rating_count
        return rating_average


    def compute_prediction(self, u, i):
        '''Obtiene la predicción de del usuario u, con sus factores latentes,
        al item i, con sus factores, pero esta vez con los bias.'''

        deviation = np.dot(self.p[u], self.q[i])
        prediction = self.rating_avg + self.bu[u] + self.bi[i] + deviation
        return prediction
    

    def train(self, num_iterations):

        for it in tqdm(range(num_iterations)):

            updated_p = self.p.copy() # clone p matrix
            updated_q = self.q.copy() # clone q matrix

            updated_bu = self.bu.copy() # clone bu vector
            updated_bi = self.bi.copy() # clone bi vector

            error_list = []

            for u in range(self.num_users):

                for i in self.get_items(u):
                
                    error = self.ratings[u][i] - self.compute_prediction(u, i)
                    error_list.append(abs(error))

                    updated_p[u] += self.learning_rate * (error * self.q[i] - self.regularization * self.p[u])
                    updated_q[i] += self.learning_rate * (error * self.p[u] - self.regularization * self.q[i])

                    updated_bu[u] += self.learning_rate * (error - self.regularization * updated_bu[u])
                    updated_bi[i] += self.learning_rate * (error - self.regularization * updated_bi[i])

            #print(f'Error: {np.mean(error)}')
            
            self.p = updated_p
            self.q = updated_q

            self.bu = updated_bu
            self.bi = updated_bi