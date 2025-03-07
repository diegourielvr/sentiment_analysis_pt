from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split
from config.constants_models import RANDOM_STATE, TEST_SIZE
from src.models.utils import get_report 

class MyDataset(Dataset):
    def __init__(self, x_sparse, y):
        """x_sparse es una matriz dispersa,
        y es un arreglo o serie de pandas
        """
        self.x = x_sparse # matriz dispersa devuelta por fit.transform()
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Convertir los datos a tensores solo cuando son necesarios
        # con toarray() convertimos el vector disperso en vector denso
        x_row = torch.tensor(
            self.x[idx].toarray(),
            dtype=torch.float32
        ).squeeze()
        y_row = self.y[idx]
        return x_row, y_row

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.h = nn.Linear(
            input_dim,
            1
        )
    def forward(self, X):
        y_pred = F.sigmoid(self.h(X))
        return y_pred


class OneVsAllClassifierLR:
    def __init__(self, input_dim, output_dim, lr=0.1):
        self.num_clases = output_dim
        self.models = [
            LogisticRegressionModel(input_dim)
            for _ in range(output_dim)
        ]
        self.optimizers = [
            torch.optim.SGD(model.parameters(), lr=lr)
            for model in self.models
        ]
        self.cost_function = nn.BCELoss() # Binary Cross Entropy Loss
    
    def train(self, x_train, y_train, epochs=50, batch_size=64):
        """x_train, y_train deben ser matrices densas,
        devueltas por fit_transform o transform y tener la misma longitud
        en la primera dimension
        """
        dataset = MyDataset(x_train, y_train)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        for epoch in tqdm(range(epochs)):
            for batch_x, batch_y in dataloader:
                # Convertir a one hot
                batch_y_ohe = torch.eye(self.num_clases)[batch_y]
                batch_y_ohe = batch_y_ohe.to(dtype=torch.float32) # BCELoss espera valores float32
                
                for i, model in enumerate(self.models):
                    # Recuperar el optimizador del modelo i
                    optimizer = self.optimizers[i]
                    y_true = batch_y_ohe[:, i].unsqueeze(1) # tomar la columna de la clase actual
                    
                    # Establecer en cero los gradientes
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = self.cost_function(outputs, y_true)
                    loss.backward() # Calcular gradientes
                    optimizer.step() # Actualizar parámetros
                    
    def predict(self, X):
        with torch.no_grad():
            # Obtener probabilidades de cada modelo
            pred = torch.cat(
                [model(X) for model in self.models],
                dim=1
            )
            # obtener la probabilidad más alto para cada dato
            return torch.argmax(pred, dim=1)
        
    def evaluate(self, x_test_sparse, y_test, batch_size=64):
        """x_test, y_test debe ser tensores
        """
        # Cargar datos de prueba por lotes, para reducir el consumo de RAM
        dataset = MyDataset(x_test_sparse, y_test)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Obtener total de datos de prueba
        total_samples = len(test_dataloader.dataset)
        y_test_np = np.empty(total_samples, dtype=np.int64)
        y_pred_np = np.empty(total_samples, dtype=np.int64)
        start_idx = 0 # indice para almacenando predicciones en cada iteración
        
        for x_test_batch, y_test_batch in test_dataloader:
            # Clacular tamaño del batch actual (puede ser menor el último batch)
            batch_size_actual = y_test_batch.shape[0]
            
            # Realizar preidcciones
            y_pred_batch = self.predict(x_test_batch)
            
            # Llenar arreglos con las predicciones
            y_test_np[start_idx:start_idx+batch_size_actual] = y_test_batch.numpy()
            y_pred_np[start_idx:start_idx+batch_size_actual] = y_pred_batch.numpy()

            # Actualizar indice para el siguiente batch
            start_idx += batch_size_actual

        # Generar reporte de metricas
        report = get_report(y_pred_np, y_test)
        return report
        
class StartModel:
    def start(self,X, y, vec='TFIDF', num_classes=3, epochs=50, batch_size=64,lr=0.1):
        """x, y son columnas de un df
        """
        
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        print(f"Tamaño de datos de entrenamiento: {x_train.shape[0]}")
        print(f"Tamaño de datos de prueba: {x_test.shape[0]}")

        if vec == 'TFIDF': vectorizer = TfidfVectorizer()
        elif vec == 'BOW': vectorizer = CountVectorizer()
        else:
            print(f"Tipo de vectorización: {vec} no reconocido")
            return None
        print(f"Tipo de vectorización de datos: {vec}")

        # Vectorizar y convertir en matrices densas
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        
        # Obtener númeo de características (tamaño del vocabulario)
        input_dim = x_train.shape[1]
        print(f"Tamaño del vocabulario: {input_dim}")
        
        # Entrenmaiento
        model = OneVsAllClassifierLR(
            input_dim,
            num_classes,
            lr=lr
        )
        
        print("Entrenando modelo...")
        model.train(
            x_train,
            y_train,
            epochs,
            batch_size,
        )
        
        print("Evaluando modelo...")
        report = model.evaluate(x_test, y_test.to_numpy(), batch_size)
        report['model'] = f"LR_{vec}"
        return report

        
        
        