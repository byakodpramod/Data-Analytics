library(car)
library(MASS)
library(jpeg)
library(ggplot2)
library(lattice)
library(dplyr)
library(data.table)
library(VIM)
library(Amelia)
library(corrplot)
library(mice)
library(car)
library(rpart)
library(caret)
library(glmnet)
library(elasticnet)
library(knitr)
library(pls)  #load the plsr package for Partial Least Squares Regression
library(plotly)
library(caret)
library(Boruta) # for feature recontruction
library(glmnet)
#2.a
housing.data <- read.csv("housingData2.csv") #loading data
housing.data <- subset(housing.data,select = -c(X.1))
housing.data.test <- read.csv("housingTest2.csv")
housing <- housing.data
dim(housing.data)
head(housing.data)
summary(housing.data)

#from the summary of the data we can say that few columns have missing values and those missing values can be categeroized among categorical and numerical values
#The missing values indicate that majority of the houses do not have alley access, no pool, no fence and no elevator, 2nd garage, shed or tennis court that is covered by the MiscFeature.

#handling missing data
colSums(sapply(housing.data.test, is.na))
housing.data.test <- subset(housing.data.test,select = -c(Alley,PoolQC,X))
housing.data.test <- mice(housing.data.test, maxit = 2)
housing.data.test <- complete(housing.data.test,2)

names(housing.data)
include <- setdiff(names(housing.data), c("Alley","PoolQC","X"))
housing.data <- housing.data[,include]
housing.data.mice <- mice(housing.data, maxit = 2)
housing.data.fill <- complete(housing.data.mice,2)
#feature selection 
# for feature construction we have bortura package which gives the imp and tentative variables
set.seed(13)
bor.results <- Boruta(housing.data.fill[,1:70],housing.data.fill$SalePrice,maxRuns=101,doTrace=0)
### Boruta results
cat("\nSummary of Boruta run:\n")
print(bor.results)

#These attributes were deemed as relevent to predicting house sale price.
cat("\n\nRelevant Attributes:\n")
getSelectedAttributes(bor.results)

# by using Boruta is an all relevant feature selection wrapper algorithm, capable of working with any classi-
#fication method that output variable importance measure (VIM); by default, Boruta uses Random
#Forest. The method performs a top-down search for relevant features by comparing original attributes'
#importance with importance achievable at random, estimated using their permuted copies,
#and progressively elliminating irrelevant featurs to stabilise that test.
#Home values are influenced by many factors. Basically, there are two major aspects:
#The environmental information, including location, local economy, school district, air quality, etc.
#The characteristics information of the property, such as lot size, house size and age, the number of rooms, heating / AC systems, garage, and so on.

#'LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF' all these variables are related to area type so we can add all those values int single one


#2.1 OLS


housing.data.validations <- housing.data.fill[1:100,]
housing.data.train <- housing.data.fill[101:1000,]
lapply(housing.data.train, as.numeric)
obs.salesprice <- housing.data.validations$SalePrice
#obs.housing.data <- as.matrix(housing.data.test[,-51])
#obs.housing.data<-cbind(intercept=1,obs.housing.data)

#squaring x
fit <- lm(housing.data.train$SalePrice~ ., housing.data.train)
#R Square is not bad, but many variables do not pass the Hypothesis Testing, so the model is not perfect. Potential overfitting will occur if someone insist on using it. Therefore, the variable selection process should be involved in model construction. I prefer to use Step AIC method.
#Selecting variable subsets requires the definition of a numerical criterion which measures the quality of any given variable subset. In a univariate multiple linear regression, for example, possible measures of the quality of a subset of predictors are the coefficient of determination R2, the FF statistic in a goodness-of-fit test, its corresponding pp-value or Akaike's Information Criterion (AIC)

stepfit <- stepAIC(fit)
summary(stepfit)
#The R Square is not bad, and all variables pass the Hypothesis Test. The diagonsis of residuals is also not bad. The diagnosis can be viewed below. and it gives best r^2 square value as 0.9501 for best fit

simplified_fit <- lm(formula = housing.data.train$SalePrice ~ Id + MSZoning + 
                       LotFrontage + LotArea + LotConfig + LandSlope + Neighborhood + 
                       Condition1 + BldgType + OverallQual + OverallCond + YearBuilt + 
                       RoofStyle + Exterior1st + ExterQual + ExterCond + Foundation + 
                       BsmtExposure + BsmtFinType1 + BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF + 
                       HeatingQC + CentralAir + Electrical + X1stFlrSF + X2ndFlrSF + 
                       LowQualFinSF + BsmtFullBath + BedroomAbvGr + KitchenAbvGr + 
                       Functional + Fireplaces + GarageType + GarageCars + GarageQual + 
                       GarageCond + PavedDrive + WoodDeckSF + OpenPorchSF + EncPorchSF + 
                       PoolArea + MiscFeature + MiscVal, data = housing.data.train)
layout(matrix(c(1,2,3,4), 2, 2, byrow = TRUE))
plot(simplified_fit)

cat("AIC values is")
AIC(simplified_fit)
cat("BIC values is")
BIC(simplified_fit)
cat("VIF values is")
vif(simplified_fit)
rmse <- stepfit$residuals^2 %>% mean %>% sqrt
rmse
lapply(housing.data.test, as.numeric)
predict(simplified_fit,newdata = housing.data.validations)

# influencePlot(simplified_fit) # shows the rows influence on residuals
plot(simplified_fit)

# pls hyper parameter tuning with train function
ctrl <- rfeControl(functions = caretFuncs,                                                      
                   method = "repeatedcv",
                   number=2, 
                   repeats=1,
                   verbose =TRUE
)

for(i in 1:ncol(housing.data)){
  if(is.factor(housing.data[,i])){
    housing.data[,i]<-as.numeric(housing.data[,i])
  }
}

pls.fit.rfe <- rfe(log(housing.data.fill$SalePrice) ~ .,data = housing.data.fill,   
                   method = "kernelpls",
                   preProcess = c("center", "scale"),
                   metric="RMSE",
                   sizes =  7,
                   tuneLength = 3,
                   rfeControl = ctrl
)
print(pls.fit.rfe)
# PLS
#Assessing feature importance with pls
# ecursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and weights are assigned to each one of them. Then, features whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached. after recursive training the above features are top features using Partial linearr regression
pls.fit<-plsr(SalePrice~., data= housing.data.train, validation="CV")
pls.pred<-predict(pls.fit, housing.data.test,ncomp=1)
pls.pred
plot(pls.pred,housing.data.validations$SalePrice)
plot(pls.fit)
rmse.pls <- pls.fit$residuals^2 %>% mean %>% sqrt
rmse.pls
cat("pls rmse value is")
print(rmse.pls)

# by considering the PLS features 
pls1.fit<-plsr(housing.data.train$SalePrice ~ OverallQual + GarageCars + GarageArea + GrLivArea + TotalBsmtSF, data = housing.data.train, method = "kernelpls",validation = "CV",3)
pls.pred1<-predict(pls1.fit, housing.data.test, ncomp = 1)
pls.pred1
plot(pls.pred,housing.data.validations$SalePrice)
plot(pls1.fit)
rmse.pls <- pls1.fit$residuals^2 %>% mean %>% sqrt
cat("pls rmse value is")
print(rmse.pls)

plot(pls1.fit, ncomp = 1, asp = 1, line = TRUE)
plot(RMSEP(pls1.fit),legendpos="topright")



summary(pls.fit)
pls.CVRMSE<-RMSEP(pls.fit,validation="CV")
str(pls.CVRMSE)
plot(pls.CVRMSE)

#2.d
#lasso
#The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.


housing.data.train1 <- housing.data.train

for(i in 1:ncol(housing.data.train1)){
  if(is.factor(housing.data.train1[,i])){
    housing.data.train1[,i]<-as.numeric(housing.data.train1[,i])
  }
}
for(i in 1:ncol(housing.data.validations)){
  if(is.factor(housing.data.validations[,i])){
    housing.data.validations[,i]<-as.numeric(housing.data.validations[,i])
  }
}

for(i in 1:ncol(housing.data.test)){
  if(is.factor(housing.data.test[,i])){
    housing.data.test[,i]<-as.numeric(housing.data.test[,i])
  }
}
X_train <- housing.data.train1[1:nrow(housing.data.train),]
X_train <- subset(X_train,select = -c(SalePrice))
y <- log1p(housing.data.train1$SalePrice)

# One thing to note here however is that the features selected are not necessarily the "correct" ones - especially since there are a lot of collinear features in this dataset. One idea to try here is run Lasso a few times on boostrapped samples and see how stable the feature selection is. here am running 5 times in order to tune the parameters

set.seed(123) 
CARET.TRAIN.CTRL <- trainControl(method="repeatedcv",
                                 number=5,
                                 repeats=5,
                                 verboseIter=FALSE)


model_lasso <- train(x = X_train,y = y,
                     method="glmnet",
                     metric="RMSE",
                     maximize=FALSE,
                     trControl=CARET.TRAIN.CTRL,
                     tuneGrid=expand.grid(alpha=1,  # Lasso regression
                                          lambda=c(1,0.1,0.05,0.01,seq(0.009,0.001,-0.001),
                                                   0.00075,0.0005,0.0001)))

#lsso performs even better so we'll just use this one to predict on the test set. Another neat thing about the Lasso is that it does feature selection for you - setting coefficients of features it deems unimportant to zero. Let's take a look at the coefficients:

coef <- data.frame(coef.name = dimnames(coef(model_lasso$finalModel,s=model_lasso$bestTune$lambda))[[1]], 
                   coef.value = matrix(coef(model_lasso$finalModel,s=model_lasso$bestTune$lambda)))

picked_features <- sum(coef$coef.value!=0)
not_picked_features <- sum(coef$coef.value==0)

cat("Lasso picked",picked_features,"variables and eliminated the other",
    not_picked_features,"variables\n")

coef <- arrange(coef,-coef.value)

# extract the top 10 and bottom 10 features
imp_coef <- rbind(head(coef,10),
                  tail(coef,10))

ggplot(imp_coef) +
  geom_bar(aes(x=reorder(coef.name,coef.value),y=coef.value),
           stat="identity") +
  ylim(-1.5,0.6) +
  coord_flip() +
  ggtitle("Coefficents in the Lasso Model") +
  theme(axis.title=element_blank())
#The most important positive feature is GrLivArea - the above ground area by area square feet. This definitely sense. Then a few other location and quality features contributed positively. Some of the negative features make less sense and would be worth looking into more - it seems like they might come from unbalanced categorical variables.
tune_value <- model_lasso$finalModel$tuneValue
print(tune_value)
preds <- predict(model_lasso,newdata=housing.data.test)
preds <- exp(predict(model_lasso,newdata=housing.data.test))
preds
#2.e
solution <- data.frame(Id=as.integer(rownames(housing.data.test)),SalePrice=preds)

write.csv(solution,"lasso_prediction.csv",row.names=FALSE)
