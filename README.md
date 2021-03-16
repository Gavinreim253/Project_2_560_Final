# Project_2_560_Final
Ensemble of Classifications


```{r}
knitr::opts_chunk$set(echo = TRUE)
```

```{r}
library(MASS)

library(caret)
```

#Uploading data
```{r}
library(mlbench)
data("BreastCancer")
```

#Getting rid of the ID column
```{r}
BreastCancer$Id <- NULL 
BCData <- BreastCancer
str(BCData)
```


#Converting variables from charactors to numbers
```{r}
BCData$Cl.thickness<-as.numeric(BCData$Cl.thickness)
BCData$Mitoses<-as.numeric(BCData$Mitoses)
BCData$Cell.size<-as.numeric(BCData$Cell.size)
BCData$Cell.shape<-as.numeric(BCData$Cell.shape)
BCData$Marg.adhesion<-as.numeric(BCData$Marg.adhesion)
BCData$Epith.c.size<-as.numeric(BCData$Epith.c.size)
BCData$Bare.nuclei<-as.numeric(BCData$Bare.nuclei)
BCData$Bl.cromatin<-as.numeric(BCData$Bl.cromatin)
BCData$Normal.nucleoli<-as.numeric(BCData$Normal.nucleoli)
```


#data exploration
```{r}
histogram(BCData$Cl.thickness)
histogram(BCData$Cell.size)
histogram(BCData$Cell.shape)
histogram(BCData$Marg.adhesion)
histogram(BCData$Epith.c.size)
histogram(BCData$Bare.nuclei)
histogram(BCData$Bl.cromatin)
histogram(BCData$Normal.nucleoli)
histogram(BCData$Mitoses)
```


```{r}
library(ggplot2)
Counts <- table(BCData$Class)
barplot(Counts, main ="Benign Vs. Malignant")
```


#During the decsion varaible into 1 and 0
```{r}
BCData$Class<-ifelse(BCData$Class=="malignant",1,0)
BCData$Class<-as.factor(BCData$Class)
```

#Imputing data/missing values
```{r}
library(mice)
library(VIM)
imputed.data <- mice(BCData, m=5, maxit = 50, method = 'pmm', seed = 500)
imputed.data$imp$Bare.nuclei
BCData <- complete(imputed.data,2)
summary(BCData$Bare.nuclei)
```


#Support Vector
```{r}
library(e1071)
mysvm <- svm(Class ~ ., BCData)
mysvm.pred <- predict(mysvm, BCData)
table(mysvm.pred, BCData$Class)
```


#Naive Bayes
```{r}
library(klaR)
mynb <- NaiveBayes(Class ~ ., BCData)
mynb.pred <- predict(mynb,BCData)
table(mynb.pred$class,BCData$Class)
```


#Neural net
```{r}
library(nnet)
set.seed(123)
mynnet <- nnet(Class ~ ., BCData, size=1)
mynnet.pred <- predict(mynnet,BCData,type="class")
table(mynnet.pred,BCData$Class)
```


#Decsion Tree
```{r}
library(rpart)
mytree <- rpart(Class ~ ., BCData)
mytree.pred <- predict(mytree,BCData,type="class")
table(mytree.pred,BCData$Class)

#Had to create new lable to override
mytree.pred2 <- mytree.pred
```

# Leave-1-Out Cross Validation (LOOCV)
```{r}
ans <- numeric(length(BCData[,1]))
for (i in 1:length(BCData[,1])) {
  mytree <- rpart(Class ~ ., BCData[-i,])
  mytree.pred <- predict(mytree,BCData[i,],type="class")
  ans[i] <- mytree.pred
}
ans <- factor(ans,labels=levels(BCData$Class))
table(ans,BCData$Class)
```
#QDA
```{r}
myqda <- qda(Class ~ ., BCData)
myqda.pred <- predict(myqda, BCData)
table(myqda.pred$class,BCData$Class)
```
#Rds
```{r}
set.seed(123)
myrda <- rda(Class ~ ., BCData)
myrda.pred <- predict(myrda, BCData)
table(myrda.pred$class,BCData$Class)
```
#Random Forest
```{r}
library(randomForest)
myrf <- randomForest(Class ~ .,BCData)
myrf.pred <- predict(myrf, BCData)
table(myrf.pred, BCData$Class)
```

#Creating Ensemble
```{r}
Ensemble <- as.data.frame(cbind(
  as.data.frame(mysvm.pred),
  mynb.pred$class,
  as.factor(mynnet.pred),
  mytree.pred2,
  myqda.pred$class,
  myrda.pred$class,
  myrf.pred,
  BCData$Class))
```

#Ensemble Performance
```{r}
Ensemble
str(Ensemble)
```

```{r}
Ensemble$mysvm.pred <- as.numeric(Ensemble$mysvm.pred)
Ensemble$`mynb.pred$class` <- as.numeric(Ensemble$`mynb.pred$class`)
Ensemble$`as.factor(mynnet.pred)` <- as.numeric(Ensemble$`as.factor(mynnet.pred)`)
Ensemble$mytree.pred2 <- as.numeric(Ensemble$mytree.pred2)
Ensemble$`myqda.pred$class` <- as.numeric(Ensemble$`myqda.pred$class`)
Ensemble$`myrda.pred$class` <- as.numeric(Ensemble$`myrda.pred$class`)
Ensemble$myrf.pred <- as.numeric(Ensemble$myrf.pred)
Ensemble$`BCData$Class` <- as.numeric(Ensemble$`BCData$Class`)

sum(Ensemble$mysvm.pred) #947  predicted 
sum(Ensemble$`mynb.pred$class`) #956 predicted
sum(Ensemble$`as.factor(mynnet.pred)`) #699 predcited
sum(Ensemble$mytree.pred2) #954 predicted
sum(Ensemble$`myqda.pred$class`) #957 predicted
sum(Ensemble$`myrda.pred$class`) #941 predicted 
sum(Ensemble$myrf.pred) #940 predicted
sum(Ensemble$`BCData$Class`) #940 predicted 



NBandclass <- Ensemble [c(2,8)]
NBandclass #over predicting 

rfandclass <- Ensemble [c(7,8)]
rfandclass #perfect prediction

#random forest is the best
