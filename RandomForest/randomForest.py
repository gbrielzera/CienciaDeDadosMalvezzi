import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

SEMENTE = random.randint(1, 100000)

print("\n==============================")
print("SEMENTE UTILIZADA")
print("==============================\n")

print(SEMENTE)

df = pd.read_csv("Heart_Attack_Data_Set.csv")

print("\n==============================")
print("PRIMEIRAS LINHAS")
print("==============================\n")

print(df.head())

print("\n==============================")
print("INFORMAÇÕES DA BASE")
print("==============================\n")

print(df.info())

X = df.drop(columns=["target"])

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEMENTE,
    stratify=y
)

parametros = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

modelo = RandomForestClassifier(
    random_state=SEMENTE
)

grid_search = GridSearchCV(
    estimator=modelo,
    param_grid=parametros,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

melhor_modelo = grid_search.best_estimator_

y_pred = melhor_modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)

precisao = precision_score(y_test, y_pred)

revocacao = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

matriz = confusion_matrix(y_test, y_pred)

print("\n==============================")
print("MELHORES HIPERPARÂMETROS")
print("==============================\n")

print(grid_search.best_params_)

print("\n==============================")
print("RESULTADOS RANDOM FOREST")
print("==============================\n")

print(f"Acurácia: {acuracia:.4f}")

print(f"Precisão: {precisao:.4f}")

print(f"Revocação: {revocacao:.4f}")

print(f"F1-Score: {f1:.4f}")

print("\n==============================")
print("MATRIZ DE CONFUSÃO")
print("==============================\n")

print(matriz)

print("\n==============================")
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("==============================\n")

print(classification_report(y_test, y_pred))

importancias = pd.DataFrame({
    "Variável": X.columns,
    "Importância": melhor_modelo.feature_importances_
})

importancias = importancias.sort_values(
    by="Importância",
    ascending=False
)

print("\n==============================")
print("IMPORTÂNCIA DAS VARIÁVEIS")
print("==============================\n")

print(importancias)

plt.figure(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Sem Ataque", "Com Ataque"]
)

disp.plot(cmap="Blues")

plt.title("Matriz de Confusão")

plt.show()

top10 = importancias.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top10["Variável"],
    top10["Importância"]
)

plt.xlabel("Importância")

plt.ylabel("Variável")

plt.title("Top 10 Variáveis Mais Importantes")

plt.gca().invert_yaxis()

plt.show()