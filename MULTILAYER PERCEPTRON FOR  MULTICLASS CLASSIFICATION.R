getwd()

library(tensorflow)
library(keras)
library(dplyr)

install_keras()
install_tensorflow()


data<-CTG %>% 
  select(LB:Tendency,NSP,-DR)

##CHANGE TO MATRIX

data<-as.matrix(data)
dimnames(data)<-NULL


## NORMALIZE

data[,1:21] <- normalize(data[,1:21])
data[,22]<-as.numeric(data[,22])-1

summary(data)



##data partition

set.seed(1807)

ind<- sample(2,nrow(data),replace = T,prob = c(0.7,0.3))
training<-data[ind==1,1:21]
test<-data[ind==2,1:21]

trainingtarget<-data[ind==1,22]
testtarget<-data[ind==2,22]


##one hot encoding

trainlabels<-to_categorical(trainingtarget)
testlabels<-to_categorical(testtarget)
print(testlabels)


##create sequential model

model<-keras_model_sequential()
model %>% 
  layer_dense(units = 8,activation = 'relu',input_shape = c(21)) %>% 
  layer_dense(units = 3,activation = 'softmax')

summary(model)


## compile


model %>% 
  compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics='accuracy')


##fit the model

history<-model %>% 
  fit(training,
      trainlabels,
      epoch=200,
      batch_size=32,
      validation_split=0.2)


plot(history)

model %>% 
  evaluate(test,testlabels)



##predictions and confusion matrix- test


prob<-model %>% 
  predict_proba(test)


pred<-model %>% 
  predict_classes(test)


table(predicted=pred,actual=testtarget)



cbind(prob,pred,testtarget)


