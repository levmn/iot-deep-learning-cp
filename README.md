## Treinamento de Redes Neurais (Keras) com Dados Tabulares

Este repositório contém as implementações dos exercícios de Classificação Multiclasse (Wine — UCI) e Regressão (California Housing — scikit-learn), com comparação a modelos do scikit-learn.

## Integrantes
- [RM558948] [Allan Brito Moreira](https://github.com/Allanbm100)
- [RM558868] [Caio Liang](https://github.com/caioliang)
- [RM98276] [Levi Magni](https://github.com/levmn)

### Arquivos
- `iot-nns-training.ipynb` - notebook pronto para executar no Google Colab;
- `iot-nns-training.py` - script python equivalente para execução local.

### Como executar no Google Colab (recomendado)
1. Acesse `https://colab.research.google.com/`;
2. Abra o arquivo `iot-nns-training.ipynb`;
3. Vá em Runtime/Executar tudo (ou Runtime/Run all) e aguarde o término.

Observação: o Colab já possui as dependências (TensorFlow/Keras, scikit-learn, NumPy, Pandas) pré-instaladas.

### Como executar localmente via `.py`
Pré-requisitos:
- python 3.10+ e `pip` instalados.

Instale as dependências:

```bash
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn tensorflow keras
```

Execute o script:

```bash
python iot-nns-training.py
```

Notas:
- Em Apple Silicon, se preferir, use `tensorflow-macos`. Caso tenha dificuldades com TensorFlow local, rode no Colab;
- O script baixa os datasets automaticamente e imprime as métricas ao final de cada experimento.

### Configurações dos modelos
- **Classificação (Wine/UCI)**:
  - Keras: 2 camadas ocultas (32 neurônios, reLU) + saída Softmax (3 classes)
  - Perda: `categorical_crossentropy`, Otimizador: `Adam`
  - Comparação: `RandomForestClassifier`
- **Regressão (California Housing)**:
  - Keras: 3 camadas ocultas (64, 32, 16 neurônios, reLU) + saída Linear (1 neurônio)
  - Perda: `mse`, Otimizador: `Adam`
  - Comparação: `LinearRegression`

### Resultados obtidos
Os resultados abaixo foram obtidos com `random_state=42` e normalização dos atributos.

#### Exercício 1 — Classificação (Wine)
- Rede neural (Keras):
  - Acurácia de teste = `1.0000`
- RandomForestClassifier:
  - Acurácia de teste = `1.0000`
- Observação: ambos os modelos atingiram 100% de acurácia neste conjunto de treino/teste.

#### Exercício 2 — Regressão (California Housing)
- Rede Neural (Keras): `MSE 0.2629` | `RMSE 0.5127` | `MAE 0.3345` | `R² 0.7994`
- LinearRegression: `MSE 0.5559` | `RMSE 0.7456` | `MAE 0.5955` | `R² 0.5758`
- Observação: a rede neural superou a regressão linear em todas as métricas.

### Datasets utilizados
- [Wine (UCI)](https://archive.ics.uci.edu/dataset/109/wine)
- [California Housing (scikit-learn)](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
