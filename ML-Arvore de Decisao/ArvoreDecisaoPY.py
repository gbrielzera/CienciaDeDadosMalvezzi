import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV


df = pd.read_csv('Heart_Attack_Data_Set.csv')

df.head()
df = df.rename(columns={
"age":"idade",
"sex":"sexo",
"cp":"tipo_dor_peito",
"trestbps":"pressao_repouso",
"chol":"colesterol",
"fbs":"glicemia_jejum",
"restecg":"ecg_repouso",
"thalach":"freq_cardiaca_max",
"exang":"angina_exercicio",
"oldpeak":"depressao_st",
"slope":"inclinacao_st",
"ca":"num_vasos",
"thal":"talassemia",
"target" :"desfecho"
})
df.head()

print(df["desfecho"].value_counts())

X = df.drop(columns=["desfecho"])
y = df["desfecho"] 

X_train, X_test, y_train, y_test = train_test_split(
X,y,
test_size=0.20,
random_state=42,
stratify=y) 

print(f"treino: {X_train.shape[0]} amostras")
print(f"teste : {X_test.shape[0]} amostras")


arvore = DecisionTreeClassifier( 
    max_depth= 5 ,
    min_samples_split=5,
    random_state = 42
)
arvore.fit(X_train,y_train) 
print(f"Num_folhas {arvore.get_n_leaves()}")


y_previsao = arvore.predict(X_test) 
y_probabilidade = arvore.predict_proba(X_test)[:,1] 

acuracia= sklearn.metrics.accuracy_score(y_test,y_previsao)
precisao= sklearn.metrics.precision_score(y_test,y_previsao)
revocacao= sklearn.metrics.recall_score(y_test,y_previsao) 
f1= sklearn.metrics.f1_score(y_test,y_previsao) 
auc= sklearn.metrics.roc_auc_score(y_test,y_probabilidade)

print(f"acuracia: {acuracia:.3f}")
print(f"precisao: {precisao:.3f}")
print(f"revocacao/recall: {revocacao:.3f}")
print(f"f1-score: {f1:.3f}")
print(f"auc-roc: {auc:.3f}")

print(sklearn.metrics.classification_report(y_test, y_previsao, target_names=["s/ataque", "c/ataque"]))


fig, ax = plt.subplots(figsize=(10,5))
mc = sklearn.metrics.confusion_matrix(y_test, y_previsao)
sklearn.metrics.ConfusionMatrixDisplay(mc,display_labels=["s/ataque", "c/ataque"]
).plot(
    ax=ax, 
    cmap="Blues", 
    colorbar=False
    )

ax.set_title("Matriz de Confusão")
plt.show()


fig, ax = plt.subplots(figsize=(10,5))
sklearn.metrics.RocCurveDisplay.from_predictions(y_test, y_probabilidade,ax=ax)
ax.set_title("Curva ROC")
plt.show()


plt.figure(figsize=(20,10))
plot_tree(
    arvore,
    feature_names=X.columns.tolist(),
    class_names=["sem ataque", "com ataque"],
    filled=True,
    rounded=True,
    fontsize=8)

plt.title("Árvore de Decisão Otimiz.")
plt.tight_layout()
plt.show()

parametros = { 
    "max_depth":[3, 4, 5, 6, 7], 
    "min_samples_split":[2, 5, 10] 
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    parametros,
    cv=5, 
    scoring="f1", 
)
grid.fit(X_train, y_train)

print("melhores parâmetros:", grid.best_params_)

arvore_otimi = grid.best_estimator_
y_previsao_otimi = arvore_otimi.predict(X_test)
y_probabilidade_otimi = arvore_otimi.predict_proba(X_test)[:, 1]

print(f"\nhiperparametros ajustados:")
print(f"  acurácia : {sklearn.metrics.accuracy_score(y_test, y_previsao_otimi)*100:.2f}%")
print(f"  f1-score : {sklearn.metrics.f1_score(y_test, y_previsao_otimi)*100:.2f}%")
print(f"  auc-roc  : {sklearn.metrics.roc_auc_score(y_test, y_probabilidade_otimi)*100:.2f}%")
plt.figure(figsize=(20,10))
plot_tree(
    arvore_otimi,
    feature_names=X.columns.tolist(),
    class_names=["sem ataque", "com ataque"],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Árvore Otimizada")
plt.show()


tabela_final = pd.DataFrame({
    "métrica": ["acurácia", "precisão", "revocação", "f1-score", "auc-roc"],
    
    "árvore de decisão": [
        f"{acuracia*100:.2f}%",
        f"{precisao*100:.2f}%",
        f"{revocacao*100:.2f}%",
        f"{f1*100:.2f}%",
        f"{auc*100:.2f}%"
    ]
})
tabela_final