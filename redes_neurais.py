from pybrain3.tools.shortcuts import buildNetwork  # Para criar a rede neural
from pybrain3.datasets import SupervisedDataSet  # Datasets/Conjunto de dados
from pybrain3.supervised.trainers import BackpropTrainer  # Algoritmo para treinamento


datasets = SupervisedDataSet(2, 1)  # Entradas e saídas

datasets.addSample((0.8, 0.4), 0.7)  # Quantidade de horas dormidas e quantidade de horas estudadas tendo tirado a nota
datasets.addSample((0.5, 0.7), 0.5)
datasets.addSample((0.1, 0.7), 0.95)

rede_neural = buildNetwork(2, 4, 1, bias=True)  # Passada arquitetura da rede (2 neuronios na camada de entrada +
# 4 neuronio na camada oculta + 1 neuronio na camada de saída)

trainer = BackpropTrainer(rede_neural, datasets)  # Treinador

for i in range(2000):  # Treinando a rede neural 2000 vezes
    print(trainer.train())

while True:
    dormiu = float(input('Dormiu quanto tempo? '))
    estudou = float(input('Estudou quanto tempo? '))
    z = rede_neural.activate((dormiu, estudou))[0] * 10
    print(f'Precisão da nota: {z}')




