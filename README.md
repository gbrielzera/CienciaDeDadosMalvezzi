# Comparação entre Naive Bayes, Árvore de Decisão e Random Forest

Projeto desenvolvido para a disciplina de Ciência de Dados / Inteligência Artificial com foco em classificação supervisionada utilizando R e Python.

O objetivo do trabalho foi comparar o desempenho dos algoritmos:

- Naive Bayes
- Árvore de Decisão
- Random Forest

aplicados na predição de doenças cardíacas utilizando uma base de dados clínica.

---

# Integrantes

- Ana Carolina Gontijo Vilela Dias
- André Vinícius Guedes Martins
- Filipe Silva da Fonseca
- Gabriel Cézar Peres Matos
- Gabriel Victor Vidal de Sales

---

# Objetivo do Projeto

Comparar algoritmos clássicos de Machine Learning supervisionado em duas linguagens diferentes:

- R
- Python

A análise foi realizada utilizando métricas de classificação e matrizes de confusão para avaliar o desempenho dos modelos em um problema de diagnóstico médico.

---

# Base de Dados

Dataset utilizado:

- Heart Attack / Heart Disease Dataset
- Fonte: Kaggle

A base contém variáveis clínicas relevantes, como:

- idade
- colesterol
- pressão arterial
- frequência cardíaca
- indicadores cardíacos

---

# Algoritmos Utilizados

## Python

- Gaussian Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier

Bibliotecas principais:

- scikit-learn
- pandas
- matplotlib
- seaborn

---

## R

- Naive Bayes
- Árvore de Decisão
- Random Forest

Bibliotecas principais:

- caret
- randomForest
- rpart
- e1071
- ggplot2
- pROC

---

# Estratégia Experimental

- Divisão treino/teste: 80% / 20%
- Amostragem estratificada
- Seed fixa para reprodutibilidade
- Validação cruzada (Cross Validation)
- Ajuste de hiperparâmetros com Grid Search

---

# Métricas Avaliadas

- Acurácia
- Precisão
- Recall (Revocação)
- F1-Score
- Matriz de Confusão
- Curva ROC
- AUC

---

# Resultados Gerais

Os experimentos mostraram que:

* Random Forest apresentou os resultados mais estáveis e robustos
* Árvore de Decisão teve maior interpretabilidade
* Naive Bayes apresentou ótimo desempenho no R utilizando Kernel Density
* Python facilitou tuning e modularização
* R apresentou excelente integração estatística via caret

---

# Conclusão

O projeto demonstrou que:

* não existe um algoritmo universalmente melhor
* a escolha depende do objetivo do problema
* interpretabilidade é tão importante quanto acurácia em cenários médicos
* pré-processamento influencia fortemente os resultados
* R e Python possuem desempenhos comparáveis quando o experimento é controlado corretamente

---

# Referências

* KUHN, M. Building Predictive Models in R Using the caret Package.
* PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python.
* BREIMAN, L. Random Forests.
* Kaggle Heart Disease Dataset.
