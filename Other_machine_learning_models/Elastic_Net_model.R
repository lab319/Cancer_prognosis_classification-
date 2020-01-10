rm(list=ls())
library(glmnet) 
library(caret)
library(pROC)
library(ROCR)
library(e1071)
library(caTools)
x<-read.table("HNSC_methy.txt",sep='\t',header=TRUE)
y<-read.table("HNSC_label.txt",sep='\t',header=TRUE)
dim(x)
dim(y)
head(x[1:5,1:5])

sum_AUC<-0
sum_ACC<-0
sum_PR_AUC<-0
sum_MCC<-0

set.seed(17)  
require(caret)  
folds <- createFolds(y$Stage,k=10)
for(i in 1:10){ 
	print("ÕÛÊý",i)   
  	x_train <- x[-folds[[i]],] 
  	x_test <- x[folds[[i]],]   

  	y_train <- y[-folds[[i]],] 
  	y_test <- y[folds[[i]],]   

	x_train<-as.matrix(x_train)
	y_train<-as.matrix(y_train)
	x_test<-as.matrix(x_test)
	y_test<-as.matrix(y_test)

	#################################################
	mydata <- data.frame(alpha = numeric(0), lambda = numeric(0), auc = numeric(0))
	for(a in 1:9){

		print(0.1*a)
		set.seed(1)
	
		CV = cv.glmnet(x_train, y=y_train, family='binomial', type.measure="auc",nfolds=3, alpha=0.1*a, nlambda=100)
		max_auc=max(CV$cvm)
		mydata[a,"alpha"]<-0.1*a
		mydata[a,"lambda"]<-CV$lambda.1se
		mydata[a,"auc"]<-max(CV$cvm)

	}
	param<-mydata[which(mydata$auc == max(mydata["auc"])),]
	param<-param[1,]
##############MCC############################
	Mcc <- function (act, pred){
  	TP <- sum(act == 1 & pred == 1)
  	TN <- sum(act == 0 & pred == 0)
  	FP <- sum(act == 0 & pred == 1)
  	FN <- sum(act == 1 & pred == 0)

  	denom <- as.double(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
  	if (any((TP+FP) == 0, (TP+FN) == 0, (TN+FP) == 0, (TN+FN) == 0)) denom <- 1
  	mcc <- ((TP*TN)-(FP*FN)) / sqrt(denom)
  	return(mcc)
	}

	#####################pr_auc#######################
	pr_auc <- function(obs, pred) {
  	xx.df <- prediction(pred, obs)
  	perf <- performance(xx.df, "prec", "rec")
  	xy <- data.frame(recall=perf@x.values[[1]], precision=perf@y.values[[1]])
  
  	xy <- subset(xy, !is.nan(xy$precision))

  	xy <- rbind(c(0, 0), xy)
  
  	res <- trapz(xy$recall, xy$precision)
  	res
	}
###############################################
	fit = glmnet(x_train, y=y_train, family='binomial', alpha=param["alpha"], lambda=param["lambda"])
	print(fit)
	
	predict<-predict(fit,x_test,type="class",s=param["lambda"])
	predictions<-predict.glmnet(fit,x_test,type="class",s=param["lambda"])
	predictions

	pred <- prediction(predictions, y_test)
	perf <- performance(pred,"tpr","fpr")

	auc <- performance(pred, "auc")@y.values    
	result<-confusionMatrix(factor(predict, levels = 1:0),factor(y_test, levels = 1:0))
	Acc<-result$overall['Accuracy']
	Sen<-result$byClass['Sensitivity']
	Spe<-result$byClass['Specificity']
	Mcc<-Mcc(y_test,predict)
	pr_auc<-pr_auc(y_test,predictions)
    
	sum_AUC = sum_AUC + auc[[1]]
	sum_ACC = sum_ACC + Acc
	sum_PR_AUC =sum_PR_AUC + pr_auc
	sum_MCC = sum_MCC + Mcc


	print(auc)
	print(Acc)
	print(pr_auc)
	print(Mcc)

}
print(round(sum_AUC/10,3))
print(round(sum_ACC/10,3))
print(round(sum_PR_AUC/10,3))
print(round(sum_MCC/10,3))

