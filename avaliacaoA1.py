import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.cm as cm 

# Função de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe Rede Neural
class NeuralNetwork:
    def _init_(self, input_size, hidden_sizes, output_size, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

    def feedforward(self, x):
        activations = [x]
        input = x
        for weight in self.weights:
            input = sigmoid(np.dot(input, weight))
            activations.append(input)
        return activations

    def backpropagation(self, activations, y_true):
        error = y_true - activations[-1]
        deltas = [error * sigmoid_derivative(activations[-1])]
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i + 1].T) * sigmoid_derivative(activations[i + 1])
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.atleast_2d(activations[i]).T.dot(np.atleast_2d(deltas[i]))

    def train(self, X, y):
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                activations = self.feedforward(xi)
                self.backpropagation(activations, yi)
            # Calcula e armazena o erro médio da época
            predictions = self.predict(X)
            loss = np.mean(np.square(y - predictions))
            self.loss_history.append(loss)

    def predict(self, X):
        return np.array([self.feedforward(xi)[-1] for xi in X])

# Dados XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
y = np.array([
    [0],
    [1],
    [1],
    [0],
])

# Combinações
combinacoes_sorteadas_novas = [
    (3, 0.01, 3000),  
    (5, 0.05, 8000),   
    (4, 0.2, 4000),    
    (6, 0.5, 2000),  
    (2, 0.1, 10000) 
]
# Armazena graficos
colors = cm.viridis(np.linspace(0, 1, len(combinacoes_sorteadas_novas)))
resultados = []

for idx, (neurons, lr, epochs) in enumerate(combinacoes_sorteadas_novas):
    print(f"Neurônios={neurons}, Taxa de Aprendizado={lr}, Épocas={epochs}")
    nn = NeuralNetwork(input_size=2, hidden_sizes=[neurons], output_size=1, learning_rate=lr, epochs=epochs)
    nn.train(X, y)
    predictions = nn.predict(X)
    predictions_rounded = np.round(predictions)
    accuracy = np.mean(predictions_rounded == y)
    loss = np.mean(np.square(y - predictions))

    resultados.append({
        'Neurônios Ocultos': neurons,
        'Taxa de Aprendizagem': lr,
        'Épocas': epochs,
        'Acurácia': round(accuracy, 2),
        'Erro Quadrático Médio': round(loss, 4)
    })

    # GRÁFICO DE CONVERGÊNCIA ESTILIZADO
    plt.figure(figsize=(8, 5))
    plt.plot(nn.loss_history, color=colors[idx], linewidth=2)
    plt.title(f'Convergência da MLP\nNeurônios: {neurons} | LR: {lr} | Épocas: {epochs}', fontsize=12)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio')
    plt.ylim(0, max(nn.loss_history) * 1.1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'grafico{neurons}_lr{lr}_e{epochs}.png')
    plt.close()

# Mostrar resultado final
df_resultados = pd.DataFrame(resultados)
print(df_resultados)