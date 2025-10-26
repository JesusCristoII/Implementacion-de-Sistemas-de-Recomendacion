import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class metrics:


    def __init__(self):
        pass
        

    def get_mae(self, predicciones, X_test):
        mae = []
        for i in range(X_test.shape[0]):
            linea = X_test.iloc[i]
            pred = predicciones.get(f'({int(linea["user_id"])}, {int(linea["item_id"])})')
            if pred is not None:
                mae.append(abs(pred - linea['rating']))
        if len(mae) > 0:
            return sum(mae) / len(mae), len(mae)
        

    def get_rmse(self, predicciones, X_test):

        rmse = []
        for i in range(X_test.shape[0]):
            linea = X_test.iloc[i]
            pred = predicciones.get(f'({int(linea["user_id"])}, {int(linea["item_id"])})')
            if pred is not None:
                rmse.append((pred - linea['rating'])**2)
        if len(rmse) > 0:
            return sum(rmse) / len(rmse), len(rmse)
        

    def precision(self, true_recomendations, pred_recomendations):
        '''Obtiene la precisión comparando las recomendaciones en el test que habría hecho nuestro modelo
        con las recomendaciones reales del conjunto de test.'''

        t_recomendations = []
        p_recomendations = []
        for k in true_recomendations.keys():
            true_recomendation = true_recomendations[str(k)]
            t_recomendations += true_recomendation
            # Añadimos varios -1 para asegurar que tienen la misma longitud.
            p_recomendation = (list(pred_recomendations[str(k)]) + [-1] * len(true_recomendation))[:len(true_recomendation)]
            p_recomendations += p_recomendation

        return round(precision_score(t_recomendations, p_recomendations, average= 'micro'), 4)
        

    def recall(self, true_recomendations: dict, pred_recomendations: dict):
        '''Obtiene el recall comparando las recomendaciones en el test que habría hecho nuestro modelo
        con las recomendaciones reales del conjunto de test.'''
        
        t_recomendations = []
        p_recomendations = []
        for k in true_recomendations.keys():
            true_recomendation = true_recomendations[str(k)]
            t_recomendations += true_recomendation
            # Añadimos varios -1 para asegurar que tienen la misma longitud.
            p_recomendation = (list(pred_recomendations[str(k)]) + [-1] * len(true_recomendation))[:len(true_recomendation)]
            p_recomendations += p_recomendation

        return round(recall_score(t_recomendations, p_recomendations, average= 'micro'), 4)
    

    def f1(self, true_recomendations: dict, pred_recomendations: dict):

        t_recomendations = []
        p_recomendations = []
        for k in true_recomendations.keys():
            true_recomendation = true_recomendations[str(k)]
            t_recomendations += true_recomendation
            # Añadimos varios -1 para asegurar que tienen la misma longitud.
            p_recomendation = (list(pred_recomendations[str(k)]) + [-1] * len(true_recomendation))[:len(true_recomendation)]
            p_recomendations += p_recomendation

        return round(f1_score(t_recomendations, p_recomendations, average= 'micro'), 4)