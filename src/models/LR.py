from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split
from config.constants_models import RANDOM_STATE, TEST_SIZE
from src.models.utils import evaluate

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
        """x_train, y_train deben ser tensores y tener la mismia longitud
        en la primera dimension
        """

        dataset = TensorDataset(x_train, y_train)
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
    def evaluate(self, x_test, y_test):
        """x_test, y_test debe ser tensores
        """
        y_pred = self.predict(x_test)
        # accuracy = (y_pred==y_test).float().mean()
        # print(f"Precisión en test: {accuracy:.4f}")

        # Convertir tensores a numpy para usar con sklearn
        y_true_np = y_test.numpy()
        y_pred_np = y_pred.numpy()

        report = evaluate(y_pred, y_test)
        return report
        

class LRModel:
    def start(self,X, y, vec='TFIDF', num_classes=3, epochs=50, batch_size=64,lr=0.1):
        """x, y son columnas de un df
        """
        
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        print(f"shape test: {x_test.shape}")
        if vec == 'TFIDF':
            vectorizer = TfidfVectorizer()
        elif vec == 'BOW':
            vectorizer = CountVectorizer()
        # Vectorizar y convertir en matrices densas
        x_train = vectorizer.fit_transform(x_train).toarray()
        x_test= vectorizer.transform(x_test).toarray()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        
        # convertir a tensores
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Get number of features
        input_dim = x_train_tensor.shape[1]
        
        # Entrenmaiento
        model = OneVsAllClassifierLR(
            input_dim,
            num_classes,
            lr=lr
        )
        
        print("Entrenando modelo...")
        model.train(
            x_train_tensor,
            y_train_tensor,
            epochs,
            batch_size,
        )
        
        print("Evaluando modelo...")
        report = model.evaluate(x_test_tensor,y_test_tensor)
        report['model'] = f"LR_{vec}"
        return report

        
        
        