library(MASS)
#install.packages('randomForest')
library(randomForest)
library(caret)
library(gbm)
library(mboost)
#install.packages('adabag')
library(adabag)
#install.packages('custom')
library(gbm)
library(car)
library(glmnet)
library(rpart)
library(ROCR)
library(dplyr)
library(e1071)
library(rattle)
library(party)
library(kernlab)
library(nnet)
library(devtools)
library(ggplot2)
library(custom)
library(recipes)
library(dplyr)
library(devtools)
library("ggbiplot","vqv")
library(scales)
library(grid)
library(xgboost)
library(VIM)
library(mice)

set.seed(123)
LoanDefault.Train <- read.csv("LoanDefault-train.csv")
LoanDefault.Test <- read.csv("LoanDefault-Test.csv")
x <- LoanDefault.Train[,c(1:24)]
y <- LoanDefault.Train[,25]
#converting the categorical levels to numeric 'N' = 0, 'Y' = 1
Y <- 1
N <- 0
y <- sapply(as.character(y), switch, 'N' = 0, 'Y' = 1, USE.NAMES = FALSE)
head(y)
View(LoanDefault.Train)
#Building User defined functions of model evaluations of binary classifier


ModelLog <- glm(data=LoanDefault.Train, Default ~. , family="binomial")
summary(ModelLog)
print(ModelLog)
plot(ModelLog)

Predy <- predict(ModelLog, LoanDefault.Train, type = 'terms')
head(Predy)

#user defined function for confusion Matrix
Conf.Matrix <- function(data,actual, prediction){
  cat('Confusion Matrix: ', '\n')
  return(with(data, table("Prediction"=Prediction, "Actual"= actual)))
}
#testing
Conf.Matrix(LoanDefault.Train,y,Prediction.RF2)

#User defined function for values of TP, FN, TN, FP
ClassMetrics <-function(actual, prediction){
 x <- ifelse(actual=='N' & prediction=='N', "TN",
      ifelse(actual=='N' & prediction=='Y', "FP",
      ifelse(actual=='Y' & prediction=='N',"FN","TP")))
 cat('Postives and Negatives: ', '\n')
 return(table(x))
}


#testing the function
ClassMetrics(y, Prediction.RF2)

#User defined function for accuracy
my.accuracy <- function(actual, prediction){
  y <- as.vector(table(prediction, actual))
  names(y) <- c("TN", "FP", "FN", "TP")
  acur <- (y["TP"]+y["TN"])/sum(y)
  cat('Accuracy: ', '\n')
  return(as.numeric(acur))
}
#testing function
my.accuracy(y, Prediction.RF2)

#user defined function for ROC Curve
#roc curve
plotROC <- function(data,actual, predicted){
  pred_perf <- prediction(predicted, data$labels)
  perf <- performance(pred_perf,"tpr","fpr")
  auc <- performance(pred_perf,"auc")
  auc <- auc@y.values[[1]]
  #ks statistic
  logit_ks <- max(perf@y.values[[1]]-perf@x.values[[1]])
  p <- which.max(perf@y.values[[1]]-perf@x.values[[1]])
  x <- seq(1,length(perf@y.values[[1]]), 1)
  df <- data.frame(x,perf@x.values[[1]],perf@y.values[[1]])
  names(df) <- c("x","y1","y2")
  g <- ggplot(df, aes(x))
  g <- g + geom_line(aes(y=y1), colour="blue")
  g <- g + geom_line(aes(y=y2), colour="green")
  g <- g+geom_segment(aes(x = p, y = perf@x.values[[1]][p], xend = p, yend = perf@y.values[[1]][p], color = "segment"))
  g <- g+ggtitle(paste0("ks distance =", logit_ks))
  print(g)

  roc.data <- data.frame(fpr=unlist(perf@x.values),
                         tpr=unlist(perf@y.values),
                         model="GLM")
  r <- ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
    geom_ribbon(alpha=0.2) +
    geom_line(aes(y=tpr)) +geom_abline(intercept = 0, slope = 1, color="red",
                                       linetype="dashed", size=1.5)
  r <- r+ggtitle(paste0("ROC Curve w/ AUC=", auc))

  print(r)
  #paste(logit_ks,":ks stat value"," ")
  #
  paste(auc,":auc value","and",logit_ks,":ks stat value","")
}

plotROC(LoanDefault.Train, y, Predy)
#confusion matrix metrics
conf_metrics <- function(confMat){
  specificity <- confMat$byClass["Specificity"] # proportion of true no values predicition
  Sensitivity <- confMat$byClass["Sensitivity"] # proportion of true yes values prediction or "Recall"
  Precision <- confMat$byClass["Precision"] # proportion of true positivesout of all predicted positives
  misclassification_rate <- (confMat$table[[2]]+confMat$table[[3]])/sum(confMat$table) # miss classification rate
  return(as.list(c(specificity, Sensitivity, Precision, "misclassification_rate" =
                     misclassification_rate)))
}
#D statistic
d.stat <- function(actual, predicted){
  pred.cls <- as.numeric(predicted > 0.5)
  hx<-data.frame(pred.cls, truth)
  hx.0 <- hx[hx$truth == 0,]
  hx.1 <- hx[hx$truth == 1,]
  d.stat <- mean(hx.1$pred.cls) - mean(hx.0$pred.cls)
  paste("D static value is",d.stat," ")
}

d.stat(Prediction.RF2, y)
#concordance discordance function
ConcCalc<-function(actual, predicted){
  # Get all actual observations and their fitted values into a frame
  fitted<-data.frame(cbind(predicted, actual))
  colnames(fitted)<-c('respvar','score')
  # Subset only ones
  ones<-fitted[fitted[,1]==1,]
  # Subset only zeros
  zeros<-fitted[fitted[,1]==0,]

  # Initialise all the values
  pairs_tested<-nrow(ones)*nrow(zeros)
  conc<-0
  disc<-0

  # Get the values in a for-loop
  for(i in 1:nrow(ones))
  {
    conc<-conc + sum(ones[i,"score"]>zeros[,"score"])
    disc<-disc + sum(ones[i,"score"]<zeros[,"score"])
  }
  # Calculate concordance, discordance and ties
  concordance<-conc/pairs_tested
  discordance<-disc/pairs_tested
  ties_perc<-(1-concordance-discordance)
  return(list("Concordance"=concordance,
              "Discordance"=discordance,
              "Tied"=ties_perc))
}


#Log Loss
LogLossBinary <- function(actual, predicted, eps = 1e-15) {
  metrics <- pmin(pmax(predicted, eps), 1-eps)-
    (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(predicted)
cat('Log Loss: ', '\n')
return(metrics)
  }
LogLossBinary(y, predic, eps = 1e-15)
#Exploration and visualization of data

#checking no of unique values in the variables
apply(LoanDefault.Train,2,function(x) length(unique(x)))
#checkin the structure of the variables
str(LoanDefault.Train)
dim(LoanDefault.Train)

#Checking for missing data
train.data <- LoanDefault.Train
aggr_plot <- aggr(LoanDefault.Train, col=c('navyblue','red'),
                  numbers=TRUE,
                  sortVars=TRUE,
                  labels=names(LoanDefault.Train),
                  cex.axis=.6,
                  gap=4,
                  ylab=c("Histogram of missing data","Pattern"))
aggr_plot
#There are no missing values

#Correlation among the attributes
train.data.temp <- LoanDefault.Train
for(i in 1:ncol(train.data.temp)){
  if(is.factor(train.data.temp[,i])){
    train.data.temp[,i]<-as.numeric(train.data.temp[,i])
  }
}
data.numerics <- cor(na.omit(train.data.temp))
corrplot:: corrplot(data.numerics, method="circle", type="lower", insig = "blank")
?corrplot
#There is a fairly strong correlation between status 1 - status 6 variable while there
#is a very strong correlation between Bill 1 - Bill 6 variable

#principal components analysis of datat
pca.loan <- prcomp(train.data.temp, center = TRUE, scale. = TRUE)

summary(pca.loan)
screeplot(pca.loan, main = 'Variance of PCs of Features')
#From the scree plot we can see that the amount of variation explained drops dramatically after the first 2 components.
#This suggests that just 2 components may be sufficient to summarise the data.

ggbiplot(pca.loan, obs.scale = 1, var.scale = 1, labels=row.names(LoanDefault.Train),
         ellipse = TRUE,
         circle = TRUE)+
  scale_color_discrete(name = '')+
  theme(legend.direction = 'horizontal',
        legend.position = 'top')+
  theme_bw()
#The ggbiplot indicates the 1st and 2nd PCs only captures 43% of the variation of our data



# for ggplot graph by renaming to meaningfull name for better look of graph
??doPlots
doPlots(LoanDefault.Train[,c(2:4,25)], fun = plotHist, ii = 1:4, ncol = 2)
train.data$Age<-cut(train.data$Age, breaks = c( 10, 30,50,100), labels = c("young", "middle","senior"))

aa <-ggplot(data=train.data,mapping = aes(x=Age,y=train.data$Limit,fill=Default),position = 'dodge') +
  ylab("Limit")+
  ggtitle('Boxplot for Age vs Limit')+
  geom_boxplot()+theme_bw()
#This indicates the loan default pattern is similar across age group the middle aged have higher Loan limit

ab <-ggplot(data=train.data, mapping = aes(x=MarriageStatus, fill=Default)) +
  ylab("Count")+
  geom_bar()+theme_bw()
#Marriage status is not an indication if someone will default on his/her loan payment

ac <-ggplot(data=train.data, mapping = aes(x=Gender, fill=Default))+
  ylab("Limit")+
  geom_bar()+theme_bw()
#indicates that even though a female may have high loan limit, loan default pattern is not gender  specific

ad <-ggplot(data=train.data, mapping = aes(x=Education,y=train.data$Limit,fill=Default)) +
  ylab("Limit")+
  geom_boxplot()+theme_bw()
#indicates that the University Grads and college student are more likely to take loan but the default pattern
#is similar to people with other education status

bb <-ggplot(data=train.data, aes(x = Education, fill = Default)) + geom_density() +
  xlab("Default Payment Status") +
  ylab("Customer Count")+theme_bw()

#Heat map ploting
train.data %>% group_by(Education,Age) %>% summarise(mn_creditlmt=mean(Limit)) -> df
bc <-ggplot(df, aes(Education, Age, fill=mn_creditlmt)) + geom_tile() + scale_fill_gradient(low="white", high="steelblue")

#Outliers
bd <- boxplot(Limit ~ Age, data=train.data, main="Limit readings across age")

grid.arrange(aa,ab,ac,ad, ncol = 2)



#Random forest approach random forest package
#finding best mtry for our random model
bestmtry <- tuneRF(LoanDefault.Train[,-25], LoanDefault.Train$Default,mtryStart = 10,
                   ntreeTry=1000,
                   stepFactor = 1.5,
                   improve = 0.0001,
                   trace=TRUE,
                   plot = TRUE,
                   doBest = TRUE,
                   nodesize = 30,
                   importance=TRUE)
bestmtry
#fitting the model
modelRandom <- randomForest(Default~., data = LoanDefault.Train,
                            mtry = 5,
                            metric = 'Accuracy',
                            importance = TRUE,
                            ntree = 1600)
modelRandom
importance(modelRandom)
varImpPlot(modelRandom)
Prediction.RF <- predict(modelRandom, LoanDefault.Test, type = 'prob')   #predicting response on test data

head(Prediction.RF)
solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.RF[,1])

write.csv(solution,"RF.Prediction1.csv",row.names=FALSE)


#using logistc regression
ModelLogistic <- glm(data=LoanDefault.Train, Default ~. , family="binomial")
summary(ModelLogistic)
print(ModelLogistic)

#plotting residuals of the model
par(mfrow=c(2,2))
plot(ModelLogistic)
par(mfrow=c(1,1))

#variace influence Factor
car::vif(ModelLogistic)

#making predictions with our model
Prediction.Logistic <- predict(ModelLogistic, LoanDefault.Test, type = 'response') #predicting response on test data

head(Prediction.Logistic)

solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.Logistic)

write.csv(solution,"Logistic.Prediction.csv",row.names=FALSE)



#Using stepwise selection of features
stepLogistic <- stepAIC(ModelLogistic)
summary(stepLogistic)

Prediction.StepLog <- predict(stepLogistic, LoanDefault.Test, type = 'response') #predicting response on test data

head(Prediction.StepLog)
solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.StepLog)

write.csv(solution,"StepLog.Prediction.csv",row.names=FALSE)


#Using Rpart Model form Caret package
# prepare resampling method
control <- trainControl(method="repeatedcv",
                        number=5,
                        repeats = 5,
                        classProbs=TRUE,
                        summaryFunction=mnLogLoss)
set.seed(123)
RpartModel <- train(Default~., data=LoanDefault.Train,
                    method="rpart",
                    metric="ROC",
                    preProc = 'BoxCox',
                    trControl=control)
# display results
summary(RpartModel)
plot(RpartModel)
print(RpartModel)

Prediction.Rpart <- predict(RpartModel, LoanDefault.Test, type = 'prob')   #predicting response on test data
head(Prediction.Rpart)

solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.Rpart[,2])

write.csv(solution,"Rpart.Prediction.csv",row.names=FALSE)



#fitting a gbm model
Control <- trainControl(method='repeatedcv', number=10,
                        repeats = 10,
                        returnResamp='none',
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)

set.seed(123)
gbmModel2 <- train(Default~., data = LoanDefault.Train,
                  method='gbm',
                  trControl=Control,
                  metric = 'ROC',
                  preProc = c('center','scale'))

summary(gbmModel)
print(gbmModel)

Prediction.gbm <- predict(gbmModel2, LoanDefault.Test, type = 'prob') #predicting response on test data

head(Prediction.gbm)
solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.gbm[,2])

write.csv(solution,"gbm6.Prediction.csv",row.names=FALSE)

#fitting glmboost model

set.seed(123)
controls <- trainControl(method='repeatedcv', number=10, #tuning parameters
                        repeats = 5,
                        returnResamp='none',
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)

glmboostModel <- train(Default~., data = LoanDefault.Train, #fitting the model
                       method = 'glmboost',
                       tuneLength = 5,
                       metric = 'ROC',
                       trControl = controls,
                       preProc = c('center', 'scale'))


summary(glmboostModel)
print(glmboostModel)
plot(glmboostModel)
confusionMatrix(predict(glmboostModel, LoanDefault.Train[,-25]), y)
Prediction.glmboost <- predict(glmboostModel, LoanDefault.Test, type = 'prob') #predicting response on test data

head(Prediction.glmboost)
solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.glmboost[,2])

write.csv(solution,"glmboost.Prediction.csv",row.names=FALSE)


#fitting Adaboost model
grid <- expand.grid(mfinal = (1:5)*5,
                    maxdepth = c(1, 5),
                    coeflearn = c("Breiman", "Freund", "Zhu"))
set.seed(123)
controls <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 5,
                         returnResamp = "all",
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary)

set.seed(123)
adaboostModel <- train(Default~., data = LoanDefault.Train,
                          method = 'AdaBoost.M1',
                          tuneGrid = grid,
                          metric = 'ROC',
                          trControl = controls,
                          preProc = 'pca')

adaboostModel

Prediction.adaboost <- predict(adaboostModel, LoanDefault.Test, type = 'prob') #predicting response on test data

head(Prediction.adaboost,15)
solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.adaboost[,2])

write.csv(solution,"adaboost2.Prediction.csv",row.names=FALSE)

?xgbTree


#extreme gradient boost approach
set.seed(123)

Controlz <- trainControl(method = "repeatedcv",
                        number = 5,
                        repeats = 5,
                        returnResamp = "all",
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary,
                        verboseIter = FALSE)
set.seed(123)
xtrmboostModel1 <-train(Default ~.,
                         data=LoanDefault.Train,
                         method="xgbTree",
                         metric = 'ROC',
                         preProcess = "BoxCox",
                         trControl = Controlz)
summary(xtrmboostModel1)
print(xtrmboostModel1)
plot(xtrmboostModel1)
confusionMatrix(predict(xtrmboostModel1, LoanDefault.Train[,-25]), y)
Prediction.xtrm1 <- predict(xtrmboostModel1, LoanDefault.Test, type = 'prob') #predicting response on test data

head(Prediction.xtrm1)
solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.xtrm1[,2])

write.csv(solution,"xtrm1.Prediction1.csv",row.names=FALSE)



xtrmboostModel2 <- train(Default ~ .,
                         data=LoanDefault.Train,
                         method="xgbTree",
                         metric = 'ROC',
                         preProcess = "pca",
                         trControl = Controlz)
summary(xtrmboostModel2)
print(xtrmboostModel2)
plot(xtrmboostModel2)

Prediction.xtrm2 <- predict(xtrmboostModel2, LoanDefault.Test, type = 'prob') #predicting response on test data

head(Prediction.xtrm2)
solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.xtrm2[,2])
write.csv(solution,"xtrm2.Prediction.csv",row.names=FALSE)




xtrmboostModel3 <- train(Default ~ .,
                         data=LoanDefault.Train,
                         method="xgbTree",
                         metric = 'ROC',
                         preProcess = c("scale", "center") ,
                         trControl = Controlz)
 summary(xtrmboostModel3)
print(xtrmboostModel3)
plot(xtrmboostModel3)

Prediction.xtrm3 <- predict(xtrmboostModel3, LoanDefault.Test, type = 'prob') #predicting response on test data

head(Prediction.xtrm3)
solution <- data.frame(ID=as.integer(rownames(LoanDefault.Test)),Default=Prediction.xtrm3[,2])
write.csv(solution,"xtrm3.Prediction2.csv",row.names=FALSE)


#

#for a perfect classifier
intervention.cost <- -200
penalty <- (-1600) #negative impact of interveneing in a false positive
benefit <- 3500 #benefits of intervening in a true positive

TP <- (100000) #i.e 100%
FP <- 0  #0%

#Hence the Benefit from intervention with perfect classifier
Expected.profitability <- intervention.cost*(TP+FP)+benefit*(TP)+penalty*(FP)
Expected.profitability

#expected profitability for a random model
rand <- sample(1:16000, 4000)
#predicting on a validation data of 4000 data
predRD <- predict(ModelLogistic, LoanDefault.Train[rand, ], type = 'response')
#confusion Matrix for the random model
table(LoanDefault.Train[rand,]$Default, predRD > 0.5) #a TP of 32.32% and FP of 67.68%

#hence
TP <- .3232*100000
FP <- .6768*100000

#Hence the Benefit from intervention with random model classifier
Expected.profitability <- intervention.cost*(TP+FP)+benefit*(TP)+penalty*(FP)
Expected.profitability

#using GBM model
predGBM <- predict(gbmModel, LoanDefault.Train[rand,])
head(predGBM)
confusionMatrix(predGBM,LoanDefault.Train[rand,]$Default) # TP = 67.96%, and FP = 32.04%

#hence
TP <- 0.6796*100000
FP <- 0.3204*100000

#Hence the Benefit from intervention using GBM model classifier
Expected.profitability <- intervention.cost*(TP+FP)+benefit*(TP)+penalty*(FP)
Expected.profitability


#Plotting Gain chart and lift curve
predic <- predict(gbmModel, LoanDefault.Train, type = 'prob')
p <- predic[order(predic$N),]
head(p)
# Creating the cumulative density
p$cumden <- cumsum(p$Y)/sum(p$Y)

# Creating the % of population
p$perpop <- (seq(nrow(p))/nrow(p))*100

#ggplot of random and choosen model

ggplot(data = p, aes(x = perpop, y = cumden))+
  geom_line(color = 'lightblue', lwd = 2)+theme_bw()+
  geom_abline(intercept = 0, slope = 0.01, color = 'grey', lwd = 2)+
  scale_y_continuous(breaks=seq(0,1,0.2))+
  scale_x_continuous(breaks=seq(0,100,20))+
  xlab('% of Population')+ylab("% of Target Captured")+
  ggtitle("Cummulative Gain Curve")
#calcualting the no of TP in each of our deciles
table(LoanDefault.Train$Default)
ratio.defaulter <- (3501/16000)
ratio.defaulter
