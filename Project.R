library(Hmisc)
library(caret)
library(magrittr)
library(doParallel)
library(randomForest)
library(Boruta)
library(MASS)
library(GGally)
library(corrplot)
library(tableplot)
library(tabplot)
library(Rtsne)
library(cluster)  
library(useful)     
library(NbClust)
library(textir)
library(plyr)
#install.packages('DiagrammeR')
library(DiagrammeR)
library(lattice)
library(gridExtra)
library(ggplot2)
library("ggbiplot","vqv")
library(readr)
#install.packages("FactoMineR")
library(FactoMineR)
#install.packages('googleVis')
library(googleVis)
library(VIM)
library(MLmetrics)
library(devtools)
library(psych)
devtools::install_github('topepo/caret/pkg/caret')
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

#No missing data. All predictors appears to be mostly categorical
aggr_plot <- aggr(train, col=c('navyblue','red'),
                  numbers=TRUE,
                  sortVars=TRUE,
                  labels=names(train),
                  cex.axis=.6,
                  gap=4,
                  ylab=c("Histogram of missing data","Pattern"))


#############################################
##     Understanding of data               ##
#############################################


#Outlier detection using cooks distant approach
mod <- lm(as.numeric(target) ~ ., data=train[,-1])
cooksd <- cooks.distance(mod)


plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  # add labels
#Observation 500, 527, 639 and 56959 are likely outliers


#Principal componenets
pca.train <- prcomp(train[, -c(1,95)], center = TRUE, scale. = TRUE)
summary(pca.train)

screeplot(pca.train, main = 'PCs of Features')

ggbiplot(pca.train, obs.scale = 2, var.scale = 2, labels=row.names(train[, -c(1,95)]),
         ellipse = TRUE,
         circle = TRUE, lwd = 5)+
  scale_color_discrete(name = '')+
  theme(legend.direction = 'horizontal',
        legend.position = 'top')+
  theme_bw()

#
ggplot(data = train[,-1], aes(x = target))+
  geom_bar(aes(y = (..count..)/sum(..count..)))+
  scale_y_continuous(labels=percent)+
  ylab("")+ theme_bw()

#kmeans
trainKM <- kmeans(scale(train[,-c(1,95)]),9, nstart=10)

clusplot(scale(train[,-c(1,95)]), trainKM$cluster, main='2D representation of the Cluster solution',
         color=TRUE, shade=TRUE,labels=2, lines=0)

#confusin matrix of clusters compared to the target
table(train[,95],trainKM$cluster)

#HIERARCHICAL CLUSTERING
#@ k = 9
d <- dist(scale(train[c(1:5000),-c(1,95)]), method = "euclidean") # Euclidean distance matrix.
H.fit <- hclust(d, method="ward.D")
plot(H.fit) # display dendogram
rect.hclust(H.fit, k=9, border="red")



#############################################
## Factor analysis before building models  ##
#############################################
a <- train[,-c(1,95)]
b <- test[, -1]
ProjectDataFactor <- rbind(a,b)

View(ProjectDataFactor)

Variance_Explained_Table_results<-PCA(ProjectDataFactor, graph=FALSE)
Variance_Explained_Table<-Variance_Explained_Table_results$eig
Variance_Explained_Table_copy<-Variance_Explained_Table

write.table(round(Variance_Explained_Table,3), file = "data_eigenvalues.csv", sep = ",", quote = FALSE, row.names = F)

row=1:nrow(Variance_Explained_Table)
name<-paste("Feature No:",row,sep="")
Variance_Explained_Table<-cbind(name,Variance_Explained_Table)
Variance_Explained_Table<-as.data.frame(Variance_Explained_Table)
colnames(Variance_Explained_Table)<-c("Components", "Eigenvalue", "Variance Explained (%)", "Cumulative Variance Explained (%)")
View(Variance_Explained_Table)

write.table(Variance_Explained_Table, file = "data_eigenvalues.csv", sep = ",", quote = FALSE, row.names = F)

eigenvalues  <- Variance_Explained_Table[,2]
df           <- cbind(as.data.frame(as.numeric(as.character(eigenvalues))), c(1:length(eigenvalues)), rep(1, length(eigenvalues)))
colnames(df) <- c("eigenvalues", "components", "abline")

#plotting eigenvalues and showing cut off points for optimal no of features
line <- ggplot(data = as.data.frame(df), aes(x = components))+
  geom_line(aes(y = abline), color = "red")+
  geom_point(aes(y = (eigenvalues)), color = "blue")+
  ggtitle(' Scree plot ')+
  xlab('Number of Components')+ ylab('Eigenvalues')+ theme_bw()

line

cum_variance  <- Variance_Explained_Table[,4]
df2           <- cbind(as.data.frame(as.numeric(as.character(cum_variance))), c(1:length(cum_variance)), rep(66, length(eigenvalues)))
colnames(df2) <- c("variance", "components", "abline")


#plot of % variaance explained
line2 <- ggplot(data = as.data.frame(df2), aes(x = components))+
  geom_line(aes(y = abline), color = "red")+
  geom_point(aes(y = (variance)), color = "blue")+
  ggtitle('Variance Explained by Features')+
  xlab('Number of Components')+ ylab('Cummulative Percentage Variance')+ theme_bw()

line2


#using component with eigenvalues > 1 and that accounts for 2/3 of the entire data variance
FA.ProjData <- factanal(ProjectDataFactor, scores = 'regression', factors = 32, rotation = "varimax")
print(FA.ProjData, digits = 2, cutoff =.3, sort=TRUE)


#interaction of features in loadings 1 and 2
load = FA.ProjData$loadings[,1:2]
plot(load, type="n") # set up plot 
+text(load,labels=names(train),cex=.7, color = 'blue') # add variable names

#interaction if features along loadings 3 and 4
load = FA.ProjData$loadings[,3:4]
plot(load, type="n")  # set up plot 
text(load,labels=names(train),cex=.7) # add variable names

#interaction if features along loadings 29 and 30
load = FA.ProjData$loadings[,29:30]
plot(load, type="n") # set up plot 
text(load,labels=names(train),cex=.7) # add variable names

#interaction if features along loadings 31 and 32
load = FA.ProjData$loadings[,31:32]
plot(load, type="n") # set up plot 
text(load,labels=names(train),cex=.7) # add variable names


NEW_ProjectData <- round(FA.ProjData$scores[,1:32,drop=F],3)
View(NEW_ProjectData)


y <- NEW_ProjectData[c(1:61878), ]
id <- as.data.frame(train$id)
target <- as.data.frame(train$target)
y <- cbind(id,y,target)
y <- rename(y, c('train$id'='id', 'train$target'='target'))

id <- as.data.frame(test$id)
x <- NEW_ProjectData[c(61879:206246), ]
x <- cbind(id,x) 
x<- rename(x, c('test$id'='id'))


#############################################
#############################################
##          building of models             ##
#############################################
#############################################
num_rows_sample <- 15000
train_sample <- train[sample(1:nrow(train), size = num_rows_sample),]
features     <- train_sample[,c(-1, -95)]

library('Rtsne')
tsne <- Rtsne(as.matrix(features), check_duplicates = FALSE, pca = TRUE,perplexity=30, theta=0.5, dims=2)

embedding <- as.data.frame(tsne$Y)
embedding$Class <- as.factor(sub("Class_", "", train_sample[,95]))

p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
  geom_point(size=1.25) +
  guides(colour = guide_legend(override.aes = list(size=3))) +
  xlab("") + ylab("") +
  ggtitle("Embedding of Products Data") +
  theme_light(base_size=14) +
  theme(strip.background = element_blank(),
        strip.text.x     = element_blank(),
        axis.text.x      = element_blank(),
        axis.text.y      = element_blank(),
        axis.ticks       = element_blank(),
        axis.line        = element_blank(),
        panel.border     = element_blank())
plot(p)
X <- sqrt(train[,2:94])
max_val <- 7
X[X>max_val] <- max_val
plots <- list()
colfunc <- colorRampPalette(c("white", "blue"))
x_ticks <- seq(20, ncol(X), by=20)
for(i in 1:9)
{
  #data points belonging to class class i
  class_data <- X[which(train[,95]==paste("Class_",i,sep='')),]
  
  #sort data by column values which have max median
  #idea: make it easier to spot possible clusters in each class
  by_col <- which.max(apply(class_data,2,FUN=median))
  class_data <- class_data[order(class_data[,by_col]), ]
  
  plots[[i]] <- levelplot(t(class_data),
                          xlab = '', 
                          ylab = '', 
                          colorkey = FALSE,
                          col.regions = colfunc(100),
                          at = seq(0, max_val, length.out=100),
                          aspect = 'fill',
                          scales = list(x=list(at=x_ticks, labels=x_ticks),
                                        y=list(draw=F)),
                          par.settings=list(layout.heights=list(top.padding=0,
                                                                bottom.padding = 0,
                                                                main.key.padding = -2),
                                            layout.widths = list(left.padding = 0,
                                                                 right.padding = 0)),
                          main = paste("Class_", i, sep=''))
}

viz = grid.arrange(plots[[1]],plots[[2]],plots[[3]],
                   plots[[4]],plots[[5]],plots[[6]],
                   plots[[7]],plots[[8]],plots[[9]], 
                   nrow=3, ncol=3)


##XGBoost
library(xgboost)
library(methods)
library(data.table)
library(magrittr)
train = train[,-1]
test = test[,-1]
# Gradient Boosted trees
cl <- makeCluster(7)
registerDoParallel(cl)
# runs for 15 minutes for 10 fold cross validation
levels(true_train$target) <- c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
bt_fit <- train(target~.-id, train_nzv, trControl = ctrl, method = "gbm", metric = "logLoss")
bt_fit
rf_af <- classPred(bt_fit,train_nzv)
y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

# Training parameters
param["eta"] <- 0.3   # Learning rate
param["max_depth"] <- 6  # Tree depth
nround = 50     # Number of trees to fit

# Train the model
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround, verbose=0)
bst
# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)


# neural network
# Runs for about 5 minutes in a neural network model with 10 fold cross validation

cl <- makeCluster(7)
registerDoParallel(cl)
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
nn_fit <- train(target~.-id,train_nzv, trControl = ctrl, method = "nnet", metric = "logLoss")
stopCluster(cl)
registerDoSEQ()
Partition_train <- createDataPartition(y=train_nzv$target,p=0.75,list = FALSE)
true_train <- train_nzv[Partition_train,]
true_test <- train_nzv[-Partition_train,]
classPred <- function (fit, test)
{
  fit_prob <- predict(fit, test, type = "prob")
  fit_pred <- predict(fit, test)
  levels(fit_pred) <-  c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
  test_output <- dummyVars(~ target,data = test, levelsOnly = TRUE)
  test_dv <- predict(test_output,test)
  print(confusionMatrix(fit_pred, test[[82]]))
  print ("Logloss is :")
  print(LogLoss(test_dv,as.matrix(fit_prob)))
}
nn_af <- classPred(nn_fit, true_test)
# making everything as log
# taking too long, more than 1 hour for 10 fold cross validation
# simply unpractical 2 hrs 30 minutes and still running
cl <- makeCluster(7)
registerDoParallel(cl)
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
svm_fit <- train(target~.-id,train_nzv, trControl = ctrl, method = "svmRadial", metric = "logLoss")
stopCluster(cl)
registerDoSEQ()

#RF and GBM
MCLogLoss <- function(data, lev = NULL, model = NULL)  {
  
  obs <- model.matrix(~data$obs - 1)
  preds <- data[, 3:ncol(data)]
  
  err = 0
  for(ob in 1:nrow(obs))
  {
    for(c in 1:ncol(preds))
    {
      p <- preds[ob, c]
      p <- min(p, 1 - 10e-15)
      p <- max(p, 10e-15)
      err = err + obs[ob, c] * log(p)
    }
  }
  
  out <- err / nrow(obs) * -1
  names(out) <- c("MCLogLoss")
  out
}

#TUNE HERE
#How much of the data to use to build the randomForest
TRAIN_SPLIT = 0.7
RF_MTRY = 9
RF_TREES = 125
GBM_IDEPTH = 4
GBM_SHRINKAGE  = 0.1
GBM_TREES = 50
library(caret)
#Prepare training\testing data, extract
#target and save test id's
train_gbm <- train[, -which(names(train)=="id")] 
target <- train_gbm$target 
train_gbm <- train_gbm[, -which(names(train_gbm)=="target")] 
id <- test$id 
test_gbm <- test[, -which(names(test)=="id")] 

#Split training data into two sets(keep class distribution)
set.seed(20739) 
trainIndex <- createDataPartition(target, p = TRAIN_SPLIT, list = TRUE, times = 1) 
allTrain <- train_gbm 
allTarget <- target 
train_gbm <- allTrain[trainIndex$Resample1, ] 
train2 <- allTrain[-trainIndex$Resample1, ] 
target <- allTarget[trainIndex$Resample1] 
target2 <- allTarget[-trainIndex$Resample1] 

#Build a randomForest using first training set
fc <- trainControl(method = "repeatedCV", 
                   number = 2, 
                   repeats = 1, 
                   verboseIter=FALSE, 
                   returnResamp="all", 
                   classProbs=TRUE) 
tGrid <- expand.grid(mtry = RF_MTRY) 
model <- train(x = train_gbm, y = target, method = "rf", 
               trControl = fc, tuneGrid = tGrid, metric = "Accuracy", ntree = RF_TREES) 
#Predict second training set, and test set using the randomForest
train2Preds <- predict(model, train2, type="prob") 
testPreds <- predict(model, test_gbm, type="prob")
model$finalModel

#Build a gbm using only the predictions of the
#randomForest on second training set
fc <- trainControl(method = "repeatedCV", 
                   number = 10, 
                   repeats = 1, 
                   verboseIter=FALSE, 
                   returnResamp="all", 
                   classProbs=TRUE,
                   summaryFunction=MCLogLoss)

tGrid_1 <- expand.grid(interaction.depth = GBM_IDEPTH, shrinkage = GBM_SHRINKAGE, n.trees = GBM_TREES) 
model2 <- train(x = train2Preds, y = target2, method = "gbm", 
                trControl = fc, tuneGrid = tGrid_1, metric = "MCLogLoss", verbose = FALSE)
model2
hist(model2$resample$MCLogLoss)

#Build submission
submit <- predict(model2, testPreds, type="prob") 
# shrink the size of submission
submit <- format(submit, digits=2, scientific = FALSE)
submit <- cbind(id=1:nrow(testPreds), submit) 
write.csv(submit, "rf_gbm.csv", row.names=FALSE)


install.packages('tfidf')
# making tf idf features
nzv <- nearZeroVar(train)
train_nzv <- train[,-c(nzv)]
test_nzv <- test[,-nzv]

train_tfidf <- tfidf(train_nzv[,-c(1,82)], normalize = TRUE)

# t-SNE dimensionality reduction and plotting them
# max iteration = 1000, mdims = 2, perplexity = 30 and theta = 0.5
tsne <- Rtsne(as.matrix(train_nzv[,-c(1,82)]), check_duplicates = FALSE, pca = FALSE, verbose =TRUE)

ggplot(as.data.frame(tsne$Y), aes(x = V1, y = V2, color = train_nzv$target))+
  geom_point(size = 1.25)+xlab("")+ylab("")+ggtitle("t-SNE for Otto Classification Problem")


# t-SNE dimensionality reduction of the tf-idf features

tsne_tfidf <- Rtsne(as.matrix(train_tfidf), check_duplicates = FALSE, pca = FALSE, verbose = TRUE)

ggplot(as.data.frame(tsne_tfidf$Y), aes(x = V1, y = V2, color = train_nzv$target))+
  geom_point(size = 1.25)+xlab("")+ylab("")+ggtitle("t-SNE with tf-df features for Otto Classification Problem")


# training neural network with tf - idf features

cl <- makeCluster(7)
registerDoParallel(cl)

ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
rf_full_fit <- train(x = train_tfidf, y = train_nzv$target, trControl = ctrl, method = "rf", metric = "logLoss",ntree = 100)
stopCluster(cl)
registerDoSEQ()

rough <- predict(rf_full_fit, test_nzv, type = "prob")
write.csv(rough,file ="rf_idf.csv")

# Naive bayes Classifier
library(klaR)
cl <- makeCluster(7)
registerDoParallel(cl)
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
nb_fit <- train(target~.-id,train_nzv, trControl = ctrl, method = "nb", metric = "logLoss")
stopCluster(cl)
registerDoSEQ()
# lets train our model in random forest with no feature engineering to set a bench mark
# 18 minutes for 10 fold cross validation

cl <- makeCluster(7)
registerDoParallel(cl)
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
rf_full_fit <- train(target~.-id,train_nzv, trControl = ctrl, method = "rf", metric = "logLoss",ntree = 100)
stopCluster(cl)
registerDoSEQ()

# feature construction
# a feature for row sum
row_sum <- list(0)
for (i in 1:nrow(train_nzv))
{
  row_sum[i] <-  sum(as.numeric(train_nzv[i,-c(ncol(train_nzv))]))
}

# a feature for row variance
row_var <- list(0)
for (i in 1:nrow(train_nzv))
{
  row_var[i] <-  var(as.integer(train_nzv[i,-c(ncol(train_nzv))]))
  
}

# a feature for counting non zero predictors
row_nonzero <- list(0)
for (i in 1:nrow(train_nzv))
{
  cnt = 0
  for (j in 1:(ncol(train_nzv)-1))
  {
    if (train_nzv[i,j] == 0)
    {
      cnt = cnt+1
    }
  }
  row_nonzero[i] <- cnt
}

# adding them all to the data frame

train_nzv$row_sum <- as.numeric(row_sum)
train_nzv$row_var <- as.numeric(row_var)
train_nzv$row_nonzero <- as.numeric(row_nonzero)
set.seed(123)
cl <- makeCluster(7)
registerDoParallel(cl)
rfProfile <- rfe(train_nzv[,-81],train_nzv[,81], rfeControl = rfeControl(functions = rfFuncs,method= "boot", number = 10))

# Boruta wrapper around rf for feature selection
# ran for 35 minutes and left all the variables as tentative attributes

br <- Boruta(target~.,data = true_train, doTrace = 2, maxRuns = 11)

# sample submission using random forest

cl <- makeCluster(7)
registerDoParallel(cl)

# train a random forest on the internal training set
set.seed(12631)

# with 100 trees runs 5 to 6 minutes while run on a 7 core in parallel

rf_after = foreach(ntree=rep(100,7), .combine=combine,
                   .multicombine=TRUE, .packages="randomForest") %dopar% {
                     randomForest(target~.,
                                  data=train_nzv, ntree=ntree, do.trace=1, importance=TRUE,
                                  replace=TRUE, forest=TRUE)
                   }

summary(rf)

rf_ensemble <- predict(rf_after,test_nzv)
nn_ensemble <- predict(nn_fit, test_nzv)



rf_af <- classPred(rf_after, true_test)

test_nzv <- test[,-nzv]

for (i in 1:81)
{
  test_nzv[,i] = log(test_nzv[,i]+1)
}

rough <- predict(nb_fit, test_nzv, type = "prob")
write.csv(rough,file ="nb_test.csv")


#Two layer stacking

set.seed(123)
stack_partition <- createDataPartition(y = train_nzv$target, p = 0.50, list =FALSE )
l1_partition <- train_nzv[stack_partition,]
l2_partition <- train_nzv[-stack_partition,]
train_tfidf <- as.data.frame(train_tfidf)
train_tfidf$target <- train_nzv$target
tfidf_partition <- createDataPartition(y = train_tfidf$target, p = 0.50, list = FALSE)
tfidf_l1 <- train_tfidf[tfidf_partition,]
tfidf_l2 <- train_tfidf[-tfidf_partition,]

# base models trained on l1 and meta features are extracted from it by predicting values on l2 partition.
# For the second level model these meta features along with the raw l2 features are given and trained with the l2 class labels.
# pick the strongest classifiers
# function that takes the model name and runs a 10 fold cross validation modeling using the caret's train function

mytrain <- function(xval, yval, mname)
{
  cl <- makeCluster(7)
  registerDoParallel(cl)
  ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
  rf_full_fit <- train(x = xval, y = yval, trControl = ctrl, method = mname, metric = "logLoss")
  stopCluster(cl)
  registerDoSEQ()
  return (rf_full_fit)
}

# Random forest, XGB abd KNN on l1 partition

rf_l1_partition <- mytrain(l1_partition[,-c(1,82)],l1_partition[,82],"rf")
xgb_l1_partition <- mytrain(l1_partition[,-c(1,82)], l1_partition[,82], "xgbTree")
knn_l1_partition <- mytrain(l1_partition[,-c(1,82)], l1_partition[,82], "knn")

# Random forest, XGB abd KNN on l1 partition of the tfidf features

rftfidf_l1_partition <- mytrain(tfidf_l1[,-c(81)],tfidf_l1[,81],"rf")
xgbtfidf_l1_partition <- mytrain(tfidf_l1[,-c(81)],tfidf_l1[,81], "xgbTree")
knntfidf_l1_partition <- mytrain(tfidf_l1[,-c(81)],tfidf_l1[,81], "knn")

# Prediction value that acts as meta features on the other partition as test data

rf_l2_prediction <- predict(rf_l1_partition, l2_partition[,-c(1,82)])
xgb_l2_prediction <- predict(xgb_l1_partition, l2_partition[,-c(1,82)])
knn_l1_partition <- predict(knn_l1_partition, l2_partition[,-c(1,82)])

# Predictions values that acts as meta features derived from the tf idf features of the train data set

rtfidf_prediction <- predict(rftfidf_l1_partition, tfidf_l2[,-81])
xgbtfidf_prediction <- predict(xgbtfidf_l1_partition, tfidf_l2[,-81])
knntfidf_prediction <- predict(knntfidf_l1_partition, tfidf_l2[,-81])


# Adding all the meta features to the l2 partition

l2_partition$rf <- rf_l2_prediction
l2_partition$xgb <- xgb_l2_prediction
l2_partition$knn <- knn_l1_partition
l2_partition$rftfidf <- rtfidf_prediction
l2_partition$xgbttfidf <- xgbtfidf_prediction
l2_partition$knntfidf <- knntfidf_prediction


# training a RF model on top of all these meta features

rf_2ndlayer <- mytrain(l2_partition[,-c(1,82)],l2_partition$target, "rf")
nn_2ndlayer <- mytrain(l2_partition[,-c(1,82)],l2_partition$target,"nnet")
xgb_2ndlayer <- mytrain(l2_partition[,-c(1,82)], l2_partition$target, "xgbTree")

## using the 2nd layer model to train the test data set
## Adding the first layer meta variables to the test data set

rf_test <- predict(rf_l1_partition, test_nzv[,-c(1)])
xgb_test <- predict(xgb_l1_partition, test_nzv[,-c(1)])
knn_test <- predict(knn_l1_partition, test_nzv[,-c(1)])
rftfidf_test <- predict(rftfidf_l1_partition, test_nzv[,-c(1)])
xgbtfidf_test <- predict(xgbtfidf_l1_partition, test_nzv[,-c(1)])
knntfidf_test <- predict(knntfidf_l1_partition, test_nzv[,-c(1)])


test_nzv$rf <- rf_test
test_nzv$xgb <- xgb_test
test_nzv$knn <- knn_test
test_nzv$rftfidf <- rftfidf_test
test_nzv$xgbttfidf <- xgbtfidf_test
test_nzv$knntfidf <- knntfidf_test

# Training the meta features using the second layer models

rf_final <- predict(rf_2ndlayer, test_nzv[,c(-1)], type="prob")
nn_final <- predict(nn_2ndlayer, test_nzv[,c(-1)], type = "prob")

nn_final

rf_nn_stack <- (rf_final+nn_final)/2
write.csv(rf_nn_stack, "rf_nn_stck.csv")