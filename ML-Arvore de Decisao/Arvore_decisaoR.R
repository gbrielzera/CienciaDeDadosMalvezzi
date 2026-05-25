library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(ggplot2)

df <- read.csv("Heart_Attack_Data_Set.csv")

colnames(df) <- c("idade","sexo","tipo_dor_peito","pressao_repouso",
                "colesterol","glicemia_jejum","ecg_repouso",
                  "freq_cardiaca_max","angina_exercicio","depressao_st",
                  "inclinacao_st","num_vasos","talassemia","desfecho")

print(table(df$desfecho))


X <- df[ , colnames(df) != "desfecho"]
y <- df$desfecho

set.seed(42)
idx     <- createDataPartition(y, p = 0.80, list = FALSE)
X_train <- X[ idx, ] ; X_test  <- X[-idx, ]
y_train <- y[ idx]   ; y_test  <- y[-idx]

cat(sprintf("treino: %d amostras\nteste : %d amostras\n",
            nrow(X_train), nrow(X_test)))



arvore <- rpart(
  desfecho ~ .,
  data    = cbind(X_train, desfecho = y_train),
  method  = "class",
  parms   = list(split = "gini"),
  control = rpart.control(maxdepth = 5, minsplit = 5, minbucket = 2, cp = 0)
)

cat(sprintf("Num_folhas: %d\n", sum(arvore$frame$var == "<leaf>")))

y_previsao      <- predict(arvore, X_test, type = "class")
y_probabilidade <- predict(arvore, X_test, type = "prob")[, "1"]

cm  <- confusionMatrix(factor(y_previsao, levels=c(0,1)),
                       factor(y_test,     levels=c(0,1)), positive = "1")
auc <- roc(y_test, y_probabilidade, quiet = TRUE)$auc

cat(sprintf("acuracia: %.3f\n", cm$overall["Accuracy"]))
cat(sprintf("precisao: %.3f\n", cm$byClass["Precision"]))
cat(sprintf("revocacao/recall: %.3f\n", cm$byClass["Recall"]))
cat(sprintf("f1-score: %.3f\n", cm$byClass["F1"]))
cat(sprintf("auc-roc: %.3f\n", auc))


mc_df <- as.data.frame(cm$table)
colnames(mc_df)<- c("Previsto","Real","Freq")
mc_df$Previsto<- factor(mc_df$Previsto, labels = c("s/ataque","c/ataque"))
mc_df$Real<- factor(mc_df$Real,     labels = c("s/ataque","c/ataque"))

ggplot(mc_df, aes(x = Previsto, y = Real, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 8, fontface = "bold") +
  scale_fill_gradient(low = "#DDEEFF", high = "#1565C0") +
  labs(title = "Matriz de Confusão", x = "Previsto", y = "Real") +
  theme_minimal(base_size = 14) + theme(legend.position = "none")


png("arvore_decisaoR.png", width = 2400, height = 1400, res = 120)

rpart.plot(
  arvore,
  type = 2,
  extra = 107,
  tweak = 1.1,
  branch = 0.5,
  box.palette = list("0" = "#F4A460", "1" = "#6BAED6"),
  split.font = 1,
  split.cex = 1.2,
  main = "Árvore de Decisão"
)
dev.off()


print(data.frame(
  metrica = c("acurácia","precisão","revocação","f1-score","auc-roc"),
  `arvore de decisão` = sprintf("%.2f%%", c(
    cm$overall["Accuracy"], cm$byClass["Precision"],
    cm$byClass["Recall"],   cm$byClass["F1"],
    as.numeric(auc)) * 100),
  check.names = FALSE
))