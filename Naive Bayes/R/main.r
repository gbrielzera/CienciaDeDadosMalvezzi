# install.packages(c("caret", "e1071", "ggplot2", "reshape2"))

library(caret)
library(ggplot2)
library(reshape2)

set.seed(42)

f <- function(...) {
    return(paste0(...))
}


df <- read.csv('heart.csv', sep = ',', strip.white = TRUE)

df$doenca_cardiaca <- as.factor(df$doenca_cardiaca)


train_index <- createDataPartition(df$doenca_cardiaca, p = 0.8, list = FALSE)

train_data <- df[train_index, ]
test_data  <- df[-train_index, ]


param_grid_nb <- expand.grid(
    fL = c(0, 1, 2),             
    usekernel = c(FALSE, TRUE),  
    adjust = c(1)               
)

train_control <- trainControl(
    method = "cv", 
    number = 5
)

grid_nb <- train(
    doenca_cardiaca ~ ., 
    data = train_data, 
    method = "nb",               
    trControl = train_control,
    tuneGrid = param_grid_nb,
    preProcess = c("center", "scale"),
    metric = "Accuracy"
)


previsoes <- predict(grid_nb, newdata = test_data)

matriz_detalhada <- confusionMatrix(previsoes, test_data$doenca_cardiaca)
matriz <- matriz_detalhada$table

acuracia_teste <- matriz_detalhada$overall["Accuracy"] * 100


matriz_melted <- melt(matriz)
colnames(matriz_melted) <- c("Previsto", "Real", "Valor")

levels(matriz_melted$Previsto) <- c("Saudável", "Doente")
levels(matriz_melted$Real) <- c("Saudável", "Doente")

ggplot(data = matriz_melted, aes(x = Previsto, y = Real, fill = Valor)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "#e3f2fd", high = "#0d47a1") + 
    geom_text(aes(label = Valor), color = "black", size = 5) +
    labs(
        title = "Matriz de Confusão: Naive Bayes (e1071 Otimizado)",
        x = "Previsto pela IA",
        y = "Realidade (Médico)"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))


cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat(sprintf("%45s\n", "RELATÓRIO TÉCNICO: NAIVE BAYES (R - e1071)"))
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\n[CONFIGURAÇÃO DO MODELO]\n")
cat(f("Melhor configuração 'fL' (Laplace) encontrada: ", grid_nb$bestTune$fL, "\n"))
cat(f("Usou estimativa Kernel? ", grid_nb$bestTune$usekernel, "\n"))
cat(f("Acurácia média na Validação Cruzada: ", round(max(grid_nb$results$Accuracy), 4), "\n"))

cat("\n[DESEMPENHO NO TESTE]\n")
cat(f("Acurácia Geral: ", round(acuracia_teste, 2), "%\n"))

cat("\nMatriz de Confusão:\n")
cat(paste(rep("-", 60), collapse = ""), "\n")

matriz_alinhada <- t(matriz)
df_cm <- as.data.frame.matrix(matriz_alinhada)
rownames(df_cm) <- c("Real: Saudável", "Real: Doente")
colnames(df_cm) <- c("Previsto: Saudável", "Previsto: Doente")
print(df_cm)
cat(paste(rep("-", 60), collapse = ""), "\n")

cat("\nRelatório de Métricas Detalhadas:\n")
metricas_vetor <- matriz_detalhada$byClass[c("Precision", "Recall", "F1")]

df_metricas <- data.frame(Valor = metricas_vetor)
rownames(df_metricas) <- c("Precisão (Precision)", "Revocação (Recall/Sensitivity)", "F1-Score")

print(df_metricas)
cat(paste(rep("=", 60), collapse = ""), "\n")