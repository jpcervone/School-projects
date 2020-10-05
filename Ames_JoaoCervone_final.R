#instalando as bibliotecas

library(AmesHousing)
library(tidyverse)
library(leaflet)
library(randomForest)
library(plotmo)
library(MASS)
library(partykit)
library(skimr)
library(rsample)
library(modeldata)
library(pROC)
library(glmnet)
library(rpart)


# Carregando o banco de dados

options(scipen=999)

ames <- make_ames() 

skim(ames)

ames2 <- make_ames() 

ames2$Sale_Price = NULL

# resultados dos modelos para comparação ----------------------------------


tbl <- tibble(metodo = c("LM", "Ridge", "Lasso", "Bagging", "Floresta Aleatória"), 
                        mse = NA)


#regrassão linear (lm)

mx_ames <- model.matrix(Sale_Price ~ ., data = ames) 

#transformando X em um dataframe

X <- as.data.frame(mx_ames)

class(X)

Y <- ames$Sale_Price

# criando um conjunto de treinamento vs teste -----------------------------------------------------

set.seed(123)

lotes <- sample(nrow(ames), size = .75*nrow(ames), replace = FALSE) 

treino <- X[lotes,]
teste <- X[-lotes,]

y_treino <- Y[lotes]
y_teste <- Y[-lotes]


# MODELO LINEAR -----------------------------------------------------------

lm_fit <- lm(y_treino ~ ., data = treino)

y_lm <- predict(lm_fit, teste)

tbl$mse[tbl$metodo == "LM"] <- mean((y_teste - y_lm)^2)


# RIGDE -------------------------------------------------------------------

ridge <- glmnet(mx_ames[lotes,], Y[lotes], alpha = 0, nlambda = 500)

# Obtendo os melhores valores para o log Lambda
plot_glmnet(ridge, lwd = 2, cex.lab = 1.3)

ames_ridge <- cv.glmnet(mx_ames[lotes,], Y[lotes], alpha = 0)

#Grafico mostrando a diferenca na mean-square error com a variaçao do lambda no modelo RIDGE
plot(ames_ridge, cex.lab = 1.3)

y_ridge <- predict(ridge, newx = mx_ames[-lotes,], s = ames_ridge$lambda.1se) 

tbl$mse[tbl$metodo == "Ridge"] <- mean((y_teste - y_ridge)^2)


# LASSO -------------------------------------------------------------------

lasso <- glmnet(mx_ames[lotes,], Y[lotes], alpha = 1, nlambda = 1000)

# Obtendo os melhores valores para o log Lambda com os graus de liberdade

plot_glmnet(lasso, lwd = 2, cex.lab = 1.3, xvar = "lambda")

am_lasso <- cv.glmnet(mx_ames[lotes,], Y[lotes], alpha = 1, 
                      lambda = lasso$lambda)

#Grafico mostrando a diferenca na mean-square error com a variaçao do lambda no modelo LASSO
plot(am_lasso, cex.lab = 1.3)

y_lasso <- predict(lasso, newx = mx_ames[-lotes,], s = am_lasso$lambda.min)

tbl$mse[tbl$metodo == "Lasso"] <- mean((Y[-lotes] - y_lasso)^2)


# BAGGING -----------------------------------------------------------------

arvore <- rpart(y_treino ~ ., data = ames2[lotes,])

arvore <- as.party(arvore)

y_arvore <- predict(arvore, ames2[-lotes,])

tbl$mse[tbl$metodo == "Bagging"] <- mean((y_arvore - y_teste)^2)



# Floresta Aleatória ----------------------------------------------------------

rf <- randomForest(y_treino ~ ., data = ames2[lotes,])

tibble(arvore = 1:length(rf$mse), 
       mse = rf$mse)

tbl$mse[tbl$metodo == "Floresta Aleatória"] <- mean((predict(rf, newdata = ames2[-lotes,]) - y_teste)^2)

#Grafico mostrando o numero ideal de arvores a serem plotadas

tibble(arvore = 1:length(rf$mse), 
       mse = rf$mse) %>% 
  ggplot(aes(arvore, mse)) + 
  geom_line(color = "#5B5FFF", size = 1.2) + 
  ylab("MSE (OOB)") + 
  xlab("Número de Árvores") + theme_bw()


# -------------------------------------------------------------------------

rf <- randomForest(y_treino ~ ., data = ames2[lotes,])
rf2  <- randomForest(Sale_Price ~ ., mtry = 2, data = ames2[lotes,])
rf8  <- randomForest(Sale_Price ~ ., mtry = 8, data = ames2[lotes,])
rf12 <- randomForest(Sale_Price ~ ., mtry = 12, data = ames2[lotes,])

resultados <- tibble(mtry = 4, arvore = 1:length(rf$mse), 
                     oob = rf$mse) %>% 
  bind_rows(tibble(mtry = 2, arvore = 1:length(rf2$mse), 
                   oob = rf2$mse)) %>%
  bind_rows(tibble(mtry = 8, arvore = 1:length(rf8$mse), 
                   oob = rf8$mse)) %>%
  bind_rows(tibble(mtry = 12, arvore = 1:length(rf12$mse), 
                   oob = rf12$mse))

resultados %>%  
  mutate(mtry = factor(mtry)) %>% 
  ggplot(aes(arvore, oob, group = mtry, color = mtry)) + 
  geom_line( size = 1.2) + 
  ylab("MSE (OOB)") + 
  xlab("Quantidade de Arvores") + 
  theme_bw()


# Grafico mostrando as variaveis mais importantes do modelo -------------------------------------------------

varImpPlot(rf, pch = 19)

# Conclusao:

# Comparando os modelos, o modelo LASSO aprensetou um erro de 0.2126, e teve o menor dos erros comparados com os outros modelos.
# Quando observamos a importancia das variaveis no modelo de random forest, podemos observar que as variaveis overall_Qual, neighborhood, Gr_Lib_Area e Exter_Qual, podemos observar que essas sao as 4 variaveis com maoir influencia no preco de uma residencia 



