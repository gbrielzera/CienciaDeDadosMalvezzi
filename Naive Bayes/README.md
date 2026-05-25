    # Resultados

| Dimensão Analisada   | Amostragem / Indicador                | Implementação Python (scikit-learn)   | Implementação R (caret + e1071)   |
|:---------------------|:--------------------------------------|:--------------------------------------|:----------------------------------|
| Divisão dos Dados    | Tamanho da Base de Teste              | 61 pacientes (20%)                    | 61 pacientes (20%)                |
| Otimização           | Melhor Hiperparâmetro                 | var_smoothing = 1.0                   | fL = 0 | usekernel = TRUE         |
| Validação Cruzada    | Acurácia Média (5-Fold CV)            | 0.8309                                | 0.8148                            |
| Desempenho Geral     | Acurácia no Teste                     | 0.7705                                | 0.9167                            |
| Matriz de Confusão   | Verdadeiros Saudáveis (TN)            | 16.0                                  | 24.0                              |
|                      | Falsos Doentes / Alarmes Falsos (FP)  | 12.0                                  | 3.0                               |
|                      | Falsos Saudáveis / Erros Ocultos (FN) | 2.0                                   | 2.0                               |
|                      | Verdadeiros Doentes (TP)              | 31.0                                  | 31.0                              |
| Métricas por Classe  | Precisão (Precision)                  | 72,00% (Classe Doente)                | 92,31% (Geral / Classe Positiva)  |
|                      | Revocação (Recall)                    | 94,00% (Classe Doente)                | 88,89% (Geral / Classe Positiva)  |
|                      | F1-Score                              | 82,00% (Classe Doente)                | 90,57% (Geral / Classe Positiva)  |


