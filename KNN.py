import numpy as np
from tqdm import tqdm
from Recomendation import Recomendation


class KNN(Recomendation):


    def __init__(self, ratings, test_ratings= None):
        super().__init__(ratings)
        
        self.ratings = ratings
        self.num_users = ratings.shape[0]
        self.num_items = ratings.shape[1]

        # Si no se pasa test raintgs la creamos vacía
        if test_ratings is None:
            test_ratings = np.array([[np.nan] * self.num_items] * self.num_users)
        self.test_ratings = test_ratings

        self.predictions = {}


    def rating_average(self, u):
        '''Calcula la media de las votaciones de un usuario.'''

        #return np.mean(self.ratings_only[u]).round(2)
        return np.mean(self.ratings[u][~np.isnan(self.ratings[u])]).round(2)


    def correlation_similarity(self,u, v):
        '''Calcula la correlación que hay entre las votaciones de dos usuarios para determinar
        la similaridad de los mismos.'''

        u_mean = self.rating_average(u)
        v_mean = self.rating_average(v)
        
        common_items = self.get_common_items(u, v)
        if len(common_items) > 0:
            num = np.dot(self.ratings[u][common_items] - u_mean, self.ratings[v][common_items] - v_mean)
            den = np.sqrt(np.sum((self.ratings[u][self.get_items(u)] - u_mean)**2)\
                        * np.sum((self.ratings[v][self.get_items(v)] - v_mean)**2))
            if den != 0:
                return num / den
            else:
                return None
        else:
            return None
        
    
    def get_all_corr_sim(self, u):
        '''Obtiene todas las correlaciones entre el resto de usuarios y el que es pasado por parámetro.'''
        return [self.correlation_similarity(u, v) if u != v else None for v in range(self.num_users)]
    

    def jmsd_similarity(self, u, v):
        '''Calcula la similaridad que hay entre dos usuarios mediante la métrica jsmd.'''
        jaccard = self.__jaccadrd(u, v)
        msd = self.__msd(u, v)
        if jaccard is None or msd is None:
            return None
        return jaccard * (1 - msd)


    def __jaccadrd(self, u, v):

        U = (~np.isnan(self.ratings[u])).sum()
        V = (~np.isnan(self.ratings[v])).sum()
        C = ((self.ratings[u] == self.ratings[v]) & (~np.isnan(self.ratings[u])) & (~np.isnan(self.ratings[v]))).sum()

        if U != 0 and V != 0:
            return C / (U + V - C)
        else:
            return None


    def __msd(self, u, v):
    
        num_votos = ((self.ratings[u] == self.ratings[v]) & (~np.isnan(self.ratings[u])) & (~np.isnan(self.ratings[v]))).sum()
        sumatorio = np.sum([(self.ratings[u][i] - self.ratings[v][i])**2 for i in self.get_common_items(u,v)])

        if num_votos != 0:
            return sumatorio / num_votos
        else:
            return None
        

    def get_all_jsmd_sim(self, u):
        '''Obtiene las distancias jsmd que hay entre el usuario pasado por parámetro y el resto.'''
        return [self.jmsd_similarity(u, v) if u != v else None for v in range(self.num_users)]



    def get_neighbors(self, u, k, similarities= None, corr= False, jsmd= True):
        '''Obtiene los "k" usuarios más similares a un usuario "u" dada una lista con las
        similaridades del resto de usuarios a este.'''
        if similarities is None:
            if corr:
                similarities = self.get_all_corr_sim(u)
            else:
                similarities = self.get_all_jsmd_sim(u)

        similarities_copy = np.array(similarities)
        similarities_copy = list(similarities_copy[similarities_copy != None])
        similarities_copy.sort(reverse= True)

        resul = []
        s = 0
        while len(resul) < k and s < len(similarities_copy):
            resul += list(np.where(similarities == similarities_copy[s])[0])
            s += 1
        
        return resul[:min(k, len(resul))]
    

    def average_prediction(self, u, i, k, neighbors= None, corr= False, jsmd= True):
        '''Obtiene la predicción del rating de un usario "u" a un ítem "i" mediante
        el cálculo de la media de las votaciones de los vecinos más similares
        a este usuario.'''
        if neighbors is None:
            if corr:
                neighbors = np.array(self.get_neighbors(u, k, self.get_all_corr_sim(u)))
            else:
                neighbors = np.array(self.get_neighbors(u, k, self.get_all_jsmd_sim(u)))
        
        if len(neighbors) == 0:
            return None
        
        acc = 0
        count = 0
        for n in neighbors:

            if not np.isnan(self.ratings[n][i]):
                acc += self.ratings[n][i]
                count += 1
        
        if count > 0:
            return acc / count
        else:
            return None
        
    
    def weighted_average_prediction(self, u, i, k, neighbors= None, similarities= None, corr= False, jsmd= True):
        '''Obtiene la predicción del rating de un usario "u" a un ítem "i" mediante
        la media ponderada por la similaridad de los ratings de los vecinos al item dado.'''
        if neighbors is None:
            if corr:
                similarities = np.array(self.get_all_corr_sim(u))
            else:
                similarities = np.array(self.get_all_jsmd_sim(u))
            neighbors = np.array(self.get_neighbors(u, k, similarities, corr, jsmd))
        
        sumatorio = [similarities[n] * self.ratings[n][i] for n in neighbors\
                       if not np.isnan(self.ratings[n][i])]
        
        if len(neighbors) > 0 and len(sumatorio) > 0:
            return np.sum(sumatorio) / np.sum(similarities[neighbors])
        else:
            return None
    

    def deviation_from_mean_prediction(self, u, i, k, neighbors= None, corr= False, jsmd= True):
        '''Obtiene la predicción del rating de un usario "u" a un ítem "i" mediante
        la desviación respecto a la media de los vecinos más similares al usario dado.'''
        if neighbors is None:
            if corr:
                similarities = self.get_all_corr_sim(u)
            else:
                similarities = self.get_all_jsmd_sim(u)

            neighbors = self.get_neighbors(u, k, similarities, corr, jsmd)

        sumatorio = np.array([self.ratings[n][i] - self.rating_average(n) for n in neighbors if not np.isnan(self.ratings[n][i])])
        if len(sumatorio) > 0:
            return self.rating_average(u) + (sumatorio.sum() / len(sumatorio))
        else:
            None


    def get_user_recommendations(self, u, N, k, test= False, items= None, predictions= None, ap= False, wap= True, dfmp= False, corr= False, jsmd= True):
        '''Obtiene los "N" mejores items para el usuario dado.'''

        if items is None:
            items = range(self.num_items)
       
        if predictions is None:
            if corr:
                similarities = np.array(self.get_all_corr_sim(u))
            else:
                similarities = np.array(self.get_all_jsmd_sim(u))
            neighbors = np.array(self.get_neighbors(u, k, similarities= similarities, corr= corr, jsmd= jsmd))
            if ap:
                predictions = np.array([self.average_prediction(u, i, k, neighbors) for i in items])
            elif dfmp:
                predictions = np.array([self.deviation_from_mean_prediction(u, i, k, neighbors) for i in items])
            else:
                predictions = np.array([self.weighted_average_prediction(u, i, k, neighbors, similarities) for i in items])
        
        predictions_copy = list(predictions[predictions != None])

        # Devolvemos -1 N veces porque no hay ningún item que sea -1
        if predictions_copy == []:
            return [-1] * N
        
        predictions_copy.sort(reverse= True)
        resul = []
        p = 0
        while len(resul) < N and p < len(predictions_copy):
            resul += list(np.array(items)[np.where(predictions == predictions_copy[p])[0]])
            p += 1

        # Actualizamos el diccionario con las predicciones
        for i in range(len(items)):
            self.predictions[(u, items[i])] = predictions[i]
        
        return resul[:N]
    

    def get_recomendations(self, users= None, test= False, pred= False, N= 5, k= 25, ap= False, wap= True, dpmf= False, corr= False, jsmd= True):
        '''Obtiene todas las recomendaciones de todos los usuarios.'''
        
        if test:
            return {str(u) : self.get_user_recommendations(u, N, k, test= test, items= self.get_items(u, test= pred), predictions= self.test_ratings[u][~np.isnan(self.test_ratings[u])], ap= ap, wap= wap, dfmp= dpmf, corr= corr, jsmd= jsmd)\
                    for u in tqdm(users)}
        if pred:
            return {str(u) : self.get_user_recommendations(u, N, k, items= self.get_items(u, test= pred), ap= ap, wap= wap, dfmp= dpmf, corr= corr, jsmd= jsmd)\
                    for u in tqdm(users)}
        return {str(u) : self.get_user_recommendations(u, N, k ,ap= ap, wap= wap, dfmp= dpmf, corr= corr, jsmd= jsmd) for u in tqdm(range(self.num_users))}