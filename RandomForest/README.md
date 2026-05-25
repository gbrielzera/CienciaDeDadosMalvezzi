# Random Forest - Heart Attack Dataset

Projeto desenvolvido para a disciplina de Inteligencia Artificial, utilizando o algoritmo Random Forest para classificação supervisionada em Python.

---

# Objetivo

O objetivo do projeto é prever a possibilidade de ataque cardíaco com base em atributos clínicos e médicos presentes no dataset.

O sistema realiza:

- carregamento da base de dados
- pré-processamento
- separação treino/teste
- treinamento do modelo
- ajuste de hiperparâmetros
- avaliação das métricas
- geração de gráficos
- análise de importância das variáveis

---

# Estrutura Esperada

Todos os arquivos devem estar na mesma pasta:

```text
projeto/
│
├── Heart_Attack_Data_Set.csv
├── random_forest.py
├── README.md
```

---

# Requisitos

- Python 3.10+
- pip
- ambiente virtual (recomendado)

---

# Criação do Ambiente Virtual

## Linux / macOS

```bash
python -m venv venv
```

ou

```bash
python3 -m venv venv
```

---

## Windows

```powershell
python -m venv venv
```

---

# Ativação do Ambiente Virtual

## Linux / macOS

```bash
source venv/bin/activate
```

---

## Windows CMD

```cmd
venv\Scripts\activate
```

---

## Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

---

# Instalação das Dependências

Instale as bibliotecas necessárias:

```bash
pip install pandas scikit-learn matplotlib numpy
```

---

# Dataset

O arquivo do dataset deve se chamar:

```text
Heart_Attack_Data_Set.csv
```

e deve permanecer na mesma pasta do código.

---

# Execução

Execute o projeto com:

```bash
python random_forest.py
```

ou:

```bash
py random_forest.py
```

---

# Funcionalidades do Código

O sistema executa automaticamente:

- leitura da base de dados
- separação das variáveis
- balanceamento estratificado
- geração de seed aleatória
- treinamento do modelo Random Forest
- ajuste automático de hiperparâmetros
- cálculo das métricas
- geração da matriz de confusão
- análise de importância das variáveis

---

# Métricas Geradas

O código calcula:

- Acurácia
- Precisão
- Revocação (Recall)
- F1-Score
- Matriz de Confusão

---

# Bibliotecas Utilizadas

| Biblioteca | Finalidade |
|---|---|
| pandas | Manipulação da base |
| scikit-learn | Machine Learning |
| matplotlib | Visualização gráfica |
| numpy | Operações numéricas |

---

# Hiperparâmetros Testados

O GridSearchCV realiza busca automática utilizando:

- número de árvores
- profundidade máxima
- divisão mínima de nós
- quantidade mínima de amostras por folha

---

# Reprodutibilidade

O código gera e exibe uma seed aleatória utilizada na execução.

Isso permite reproduzir os resultados posteriormente.

---

# Saídas Geradas

Durante a execução serão exibidos:

- melhores hiperparâmetros
- métricas completas
- matriz de confusão
- relatório de classificação
- gráfico da matriz de confusão
- gráfico das variáveis mais importantes

---

# Dataset Utilizado

Heart Attack Dataset

O dataset contém informações clínicas utilizadas para prever risco de ataque cardíaco.

---

# Algoritmo Utilizado

Random Forest

O Random Forest é um algoritmo de aprendizado supervisionado baseado em múltiplas árvores de decisão.

---

# Observações

- O projeto utiliza classificação supervisionada binária.
- A variável alvo utilizada é `target`.
- O sistema utiliza validação cruzada durante o ajuste de hiperparâmetros.

---

# Autor
Filipe Silva da Fonseca e Grupo

Projeto acadêmico desenvolvido para fins educacionais.
