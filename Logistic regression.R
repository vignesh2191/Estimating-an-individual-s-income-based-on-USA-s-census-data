getwd()
setwd("C:\\Users\\Admin\\Desktop\\R\\Project")
library(e1071)
library(ROCR)
data1 <- read.table("adult.data.txt",sep = ",",stringsAsFactors = TRUE)
colnames(data1) <- c("age","workclass","fnlwgt","education","education_num",
                     "marital_status","occupation","relationship","race","sex","capital_gain",
                     "capital_loss","hours_per_week","native_country","Income_Category")
trainindex<-sample(1:(nrow(data1)-1),0.6*nrow(data1))
traindata<-data1[trainindex,]
testdata<-data1[-trainindex,]
str(traindata)
str(data1)
data1$education_num <- as.numeric(data1$education_num)
data1[11:13] <- lapply(data1[11:13],as.numeric)
summary(traindata)
View(traindata)
formula_rf <- "Income_Category ~ ."
formula_ran <- as.formula(formula_rf)
traindata <- as.data.frame(traindata)

logi_model <- glm(Income_Category ~ age + workclass + fnlwgt + education + education_num
                  + marital_status + occupation + relationship + race + sex + capital_gain
                  + capital_loss + hours_per_week + native_country,family = binomial,data = traindata)
summary(logi_model)

traindata["Income_binary"] <- NA
traindata$Income_binary <- ifelse(traindata$Income_Category == " <=50K",1,0)
testdata["Income_binary"] <- NA
testdata$Income_binary <- ifelse(testdata$Income_Category == " <=50K",1,0)
traindata["native_country_binned"] <-NA
traindata$native_country_binned <- ifelse(traindata$native_country ==" United-States","US","Non-US")
testdata["native_country_binned"] <-NA
testdata$native_country_binned <- ifelse(testdata$native_country ==" United-States","US","Non-US")
View(traindata)

logi_model1 <- glm(Income_binary ~ age+workclass+fnlwgt+education_num+marital_status+occupation+relationship+race+sex+capital_gain+
                     capital_loss+hours_per_week+native_country_binned,family = binomial,data = traindata)
summary(logi_model1)

logi_model2 <- glm(Income_binary ~ age+workclass+fnlwgt+education_num+marital_status+occupation+sex+
                     +hours_per_week+native_country_binned,family = binomial(link = "logit"),data = traindata)
summary(logi_model2)

anova(logi_model1,logi_model2)
testdata$logi_model1.yhat <- predict(logi_model1,testdata,type = "response")
model <- table(testdata$Income_binary,round(testdata$logi_model1.yhat,0))
confusionMatrix(model)
pred1 <- prediction(as.numeric(testdata$logi_model1.yhat),as.numeric(testdata$Income_binary))
perf1 <- performance(pred1 , "tpr", "fpr")
plot(perf1,col="black",lty=1, lwd=2,
     main="Logistic Regrssion", cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")
abline(0,1, col="red")
auc <- performance(pred,"auc")
auc <- unlist(slot(auc, "y.values"))
auc <- round(auc,2)
legend(0.4,0.4,legend=c(paste0("AUC: ",auc)),cex=0.6,bty = "n",box.col = "white")
