# These packages Required:

library(caret)
library(ggplot2)
library(MASS)
library(car)
library(mlogit)
library(sqldf)
library(Hmisc)
library(aod)
library(BaylorEdPsych)
library(ResourceSelection)
library(pROC)
library(ROCR)
library(caTools)



#setting the working directory:

setwd("E:\\Assignment\\semester2\\machine learning\\linear")

# Reading data; just change the file path to fetch the data.

data <- read.csv("heart.csv")

##see the dataset(first 4 rows)

head(data)



# Data sanity check(structure of data)

str(data)





# Converting necessary variables into factor

data$cp <- as.factor(data$cp)
data$thal <- as.factor(data$thal)
data$slope <- as.factor(data$slope)
data$sex<-as.factor(data$sex)
data$fbs<-as.factor(data$fbs)
data$restecg<-as.factor(data$restecg)
data$exang<-as.factor(data$exang)
data$slope<-as.factor(data$slope)



## Descriptive analysis

summary(data)




## Check the missing value (if any)

sapply(data, function(x) sum(is.na(x)))



##names of the columns:

names(data)



#spliting of data:
## splitting the data into 80:20 ratio:


set.seed(122)
spl = sample.split(data$target, 0.8)
data.train = subset(data, spl == TRUE)
str(data.train)
dim(data.train)
data.test = subset(data, spl == FALSE)
dim(data.test)



## Logistic Model on Training Dataset:



model <- glm(target~sex+cp+trestbps+chol+fbs+restecg+thalach+exang+oldpeak+slope+ca+thal, data=data.train, family=binomial())
summary(model)

model <- glm(target~sex+cp+trestbps+fbs+restecg+thalach+exang+oldpeak+slope+ca, data=data.train, family=binomial())
summary(model)


model <- glm(target~sex+cp+fbs+restecg+thalach+exang+oldpeak+ca, data=data.train, family=binomial())
summary(model)


model <- glm(target~sex+cp+restecg+thalach+exang+oldpeak+ca, data=data.train, family=binomial())
summary(model)


model <- glm(target~sex+cp+trestbps+I(restecg=="1")+thalach+exang+oldpeak, data=data.train, family=binomial())
summary(model)


##Checking the multicollinearity:

vif(model)



## R square checking for the model:

# R square (nagelkarke)


modelChi <- model$null.deviance - model$deviance
#Finding the degree of freedom for Null model and model with variables
chidf <- model$df.null - model$df.residual
chisq.prob <- 1 - pchisq(modelChi, chidf)
R2.hl<-modelChi/model$null.deviance
R.cs <- 1 - exp ((model$deviance - model$null.deviance) /nrow(data))
R.n <- R.cs /(1-(exp(-(model$null.deviance/(nrow(data))))))
R.n ## ranges from 0 to 1; closer to 1 better the model



#####################################################################################################################



#checking the AUC,GINI,CONFUSION MATRIX,ACCURACY on training dataset


data.train$target<-as.factor(data.train$target)
# Predicted Probabilities
prediction <- predict(model,newdata = data.train,type="response")
library(pROC)
rocCurve   <- roc(response = data.train$target, predictor = prediction, 
levels = rev(levels(data.train$target)))
data.train$target <- as.factor(data.train$target)
#Metrics - Fit Statistics
predclass <-ifelse(prediction>coords(rocCurve,"best")[1],1,0)
Confusion <- table(Predicted = predclass,Actual = data.train$target)
AccuracyRate <- sum(diag(Confusion))/sum(Confusion)
Gini <-2*auc(rocCurve)-1
AUCmetric <- data.frame(c(coords(rocCurve,"best"),AUC=auc(rocCurve),AccuracyRate=AccuracyRate,Gini=Gini))
AUCmetric <- data.frame(rownames(AUCmetric),AUCmetric)
rownames(AUCmetric) <-NULL
names(AUCmetric) <- c("Metric","Values")
AUCmetric



##Confusion Matrix:

Confusion 

##Plotting roc curve:

plot(rocCurve)




## Since it is heart disease dataset so the error rate of heart disease patient shouldn't
be ignored:

## so the false positive rate should be minimum as possible:

## we need to change the threshold value to low:


m_or_r <- ifelse(prediction > 0.22, 1, 0)
# Convert to factor: p_class
p_class <- factor(m_or_r, levels = levels(data.train[["target"]]))
# Create confusion matrix
confusionMatrix(p_class, data.train[["target"]])

#########################################################################################################################


### KS statistics calculation



data.train$m1.yhat <- predict(model, data.train, type = "response")
library(ROCR)
m1.scores <- prediction(data.train$m1.yhat, data.train$target)
plot(performance(m1.scores, "tpr", "fpr"), col = "red")
abline(0,1, lty = 8, col = "grey")
m1.perf <- performance(m1.scores, "tpr", "fpr")
ks1.logit <- max(attr(m1.perf, "y.values")[[1]] - (attr(m1.perf, "x.values")[[1]]))
ks1.logit # Thumb rule : should lie between 40% - 70%

############################################################################################################



names(data)[9] <- "pred"
write.csv(data.train,"result.csv")


############################################################################################################


## Checking the model on testing dataset:


## Checking the AUC,GINI,Confusion Matrix,Accuracy of Testing Dataset:


data.test$target<-as.factor(data.test$target)
# Predicted Probabilities
prediction <- predict(model,newdata = data.test,type="response")
library(pROC)
rocCurve   <- roc(response = data.test$target, predictor = prediction, 
levels = rev(levels(data.test$target)))
data.test$target <- as.factor(data.test$target)
#Metrics - Fit Statistics
predclass <-ifelse(prediction>coords(rocCurve,"best")[1],1,0)
Confusion <- table(Predicted = predclass,Actual = data.test$target)
AccuracyRate <- sum(diag(Confusion))/sum(Confusion)
Gini <-2*auc(rocCurve)-1
AUCmetric <- data.frame(c(coords(rocCurve,"best"),AUC=auc(rocCurve),AccuracyRate=AccuracyRate,Gini=Gini))
AUCmetric <- data.frame(rownames(AUCmetric),AUCmetric)
rownames(AUCmetric) <-NULL
names(AUCmetric) <- c("Metric","Values")
AUCmetric
Confusion



## Lowering the False Postive Rate of Testing Dataset:

m_or_r <- ifelse(prediction > 0.22, 1, 0)
# Convert to factor: p_class
p_class <- factor(m_or_r, levels = levels(data.test[["target"]]))
# Create confusion matrix
confusionMatrix(p_class, data.test[["target"]])


############################################################################################################

##KS Stat for Testing Dataset:

data.test$m1.yhat <- predict(model, data.test, type = "response")
library(ROCR)
m1.scores <- prediction(data.test$m1.yhat, data.test$target)
plot(performance(m1.scores, "tpr", "fpr"), col = "red")
abline(0,1, lty = 8, col = "grey")
m1.perf <- performance(m1.scores, "tpr", "fpr")
ks1.logit <- max(attr(m1.perf, "y.values")[[1]] - (attr(m1.perf, "x.values")[[1]]))
ks1.logit # Thumb rule : should lie between 40% - 70%

