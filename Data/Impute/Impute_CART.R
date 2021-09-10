# impute method
setwd("~/Data/Impute")

df = read.csv("/Data/Impute/X_Before_CARTimpute.csv")

install.packages("mice")
library(mice)

col_names = names(df_new)

install.packages("randomForest")
library(randomForest)

df_imputed<- mice(df, m=1, maxit=5, method = 'cart', seed = 500)
df_imputed<-complete(df_imputed)

write.csv(df_imputed,"Data/Data.csv")
