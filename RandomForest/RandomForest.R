library(caret)
library(randomForest)
library(ggplot2)
library(pROC)

SEMENTE <- sample(1:100000, 1)

cat("\n==============================\n")
cat("SEMENTE UTILIZADA")
cat("\n==============================\n\n")

print(SEMENTE)

set.seed(SEMENTE)

df <- read.csv("Heart_Attack_Data_Set.csv")

cat("\n==============================\n")
cat("PRIMEIRAS LINHAS")
cat("\n==============================\n\n")

print(head(df))

cat("\n==============================\n")
cat("INFORMAÇÕES DA BASE")
cat("\n==============================\n\n")

str(df)

cat("\n==============================\n")
cat("VALORES AUSENTES")
cat("\n==============================\n\n")

print(colSums(is.na(df)))

df <- na.omit(df)

df$target <- as.factor(df$target)

levels(df$target) <- c(
  "Sem_Ataque",
  "Com_Ataque"
)

indices <- createDataPartition(
  df$target,
  p = 0.8,
  list = FALSE
)

train_data <- df[indices, ]

test_data <- df[-indices, ]

cat("\n==============================\n")
cat("DIVISÃO DOS DADOS")
cat("\n==============================\n\n")

cat("Treino:", nrow(train_data), "\n")

cat("Teste :", nrow(test_data), "\n")

controle <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  savePredictions = TRUE
)

grid <- expand.grid(
  mtry = c(2, 4, 6, 8)
)

modelo <- train(
  target ~ .,
  data = train_data,
  method = "rf",
  metric = "Accuracy",
  tuneGrid = grid,
  trControl = controle,
  ntree = 300,
  importance = TRUE
)

predicoes <- predict(
  modelo,
  newdata = test_data
)

probabilidades <- predict(
  modelo,
  newdata = test_data,
  type = "prob"
)

matriz <- confusionMatrix(
  predicoes,
  test_data$target
)

cat("\n==============================\n")
cat("MELHORES HIPERPARÂMETROS")
cat("\n==============================\n\n")

print(modelo$bestTune)

cat("\n==============================\n")
cat("MÉTRICAS DO MODELO")
cat("\n==============================\n\n")

acuracia <- matriz$overall["Accuracy"]

kappa <- matriz$overall["Kappa"]

precisao <- matriz$byClass["Pos Pred Value"]

recall <- matriz$byClass["Sensitivity"]

f1 <- ifelse(
  (precisao + recall) == 0,
  0,
  2 * (
    (precisao * recall) /
      (precisao + recall)
  )
)

cat("Acurácia :", round(acuracia, 4), "\n")

cat("Precisão :", round(precisao, 4), "\n")

cat("Recall   :", round(recall, 4), "\n")

cat("F1-Score :", round(f1, 4), "\n")

cat("Kappa    :", round(kappa, 4), "\n")

importancias <- importance(modelo$finalModel)

grafico_importancia <- data.frame(
  Variavel = rownames(importancias),
  Importancia = importancias[, "MeanDecreaseGini"]
)

grafico_importancia <- grafico_importancia[
  order(
    grafico_importancia$Importancia,
    decreasing = TRUE
  ),
]

cat("\n==============================\n")
cat("IMPORTÂNCIA DAS VARIÁVEIS")
cat("\n==============================\n\n")

print(grafico_importancia)

top10 <- head(grafico_importancia, 10)

grafico_importancia_plot <- ggplot(
  top10,
  aes(
    x = reorder(Variavel, Importancia),
    y = Importancia,
    fill = Importancia
  )
) +
  
  geom_bar(
    stat = "identity",
    width = 0.7
  ) +
  
  coord_flip() +
  
  labs(
    title = "Top 10 Variáveis Mais Importantes",
    x = "Variável",
    y = "Importância"
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    ),
    
    legend.position = "none"
  )

print(grafico_importancia_plot)

png(
  filename = "importancia_variaveis.png",
  width = 1200,
  height = 900,
  res = 150
)

print(grafico_importancia_plot)

dev.off()

matriz_valores <- as.matrix(matriz$table)

matriz_plot <- data.frame(
  Real = c(
    "c/ataque",
    "c/ataque",
    "s/ataque",
    "s/ataque"
  ),
  
  Previsto = c(
    "s/ataque",
    "c/ataque",
    "s/ataque",
    "c/ataque"
  ),
  
  Valor = c(
    matriz_valores[1,1],
    matriz_valores[1,2],
    matriz_valores[2,1],
    matriz_valores[2,2]
  ),
  
  Tipo = c(
    "Verdadeiro Negativo",
    "Falso Positivo",
    "Falso Negativo",
    "Verdadeiro Positivo"
  ),
  
  Cor = c(
    "Acerto",
    "Erro",
    "Erro",
    "Acerto"
  )
)

grafico_matriz <- ggplot(
  matriz_plot,
  aes(
    x = Previsto,
    y = Real,
    fill = Cor
  )
) +
  
  geom_tile(
    color = "white",
    linewidth = 2
  ) +
  
  geom_text(
    aes(
      label = paste0(
        Tipo,
        "\n\n",
        Valor
      )
    ),
    size = 7,
    fontface = "bold",
    color = "black"
  ) +
  
  scale_fill_manual(
    values = c(
      "Acerto" = "#dfeaf7",
      "Erro" = "#2f73d9"
    )
  ) +
  
  labs(
    title = "Matriz de Confusão",
    x = "Previsto",
    y = "Real"
  ) +
  
  theme_minimal(base_size = 18) +
  
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold",
      size = 30
    ),
    
    axis.title = element_text(
      face = "bold",
      size = 22
    ),
    
    axis.text = element_text(
      size = 18
    ),
    
    legend.position = "bottom",
    
    legend.title = element_blank(),
    
    panel.grid = element_blank()
  )

print(grafico_matriz)

png(
  filename = "matriz_confusao.png",
  width = 1400,
  height = 1200,
  res = 150
)

print(grafico_matriz)

dev.off()

roc_obj <- roc(
  response = test_data$target,
  predictor = probabilidades$Com_Ataque,
  levels = c(
    "Sem_Ataque",
    "Com_Ataque"
  )
)

auc_valor <- auc(roc_obj)

cat("\n==============================\n")
cat("AUC")
cat("\n==============================\n\n")

print(auc_valor)

roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

grafico_roc <- ggplot(
  roc_df,
  aes(
    x = FPR,
    y = TPR
  )
) +
  
  geom_line(
    color = "blue",
    linewidth = 1.5
  ) +
  
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "red"
  ) +
  
  labs(
    title = paste(
      "Curva ROC - AUC =",
      round(auc_valor, 4)
    ),
    
    x = "Taxa de Falsos Positivos",
    y = "Taxa de Verdadeiros Positivos"
  ) +
  
  theme_minimal(base_size = 15) +
  
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    )
  )

print(grafico_roc)

png(
  filename = "curva_roc.png",
  width = 1200,
  height = 900,
  res = 150
)

print(grafico_roc)

dev.off()
