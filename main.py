#----------------------------------------------------------------------------------------------------------
#Brecken McGeough 50301442
#ACCURACY FOR LOGISTIC REGRESSION MODEL (BATCH GRADIENT DESCENT) ON TESTING DATA: 79.87%
#ACCURACY FOR LOGISTIC REGRESSION MODEL (BATCH GRADIENT DESCENT) ON TRAINING DATA: 77.85%
#ACCURACY FOR NEURAL NETWORK (SGD) ON TESTING DATA: 79.87%
#ACCURACY FOR NEURAL NETWORK (SGD) ON TRAINING DATA: 75.57&
#Closeness of testing and training data accuracy for both models implies neither model is overfitting/underfitting
#NN constructed with 2 hidden layers, each 9 neurons deep
#----------------------------------------------------------------------------------------------------------


#Importing all the required packages
import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale,normalize
from sklearn.neural_network import MLPClassifier
import random 
from sklearn.metrics import accuracy_score


#Loading data 
df_data = pd.read_csv(r'diabetes.csv')
data_x = df_data.iloc[:,0:8]
#data_x.insert(0,"bias",[1 for _ in range(len(data_x))])
data_y =  df_data.iloc[:,8]



#Y = np_utils.to_categorical(data_y)
#standardization of independent variables
X = scale(data_x)
#Split the data into testing and training test

df = pd.DataFrame(X,columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
df.insert(0,"bias",[1 for _ in range(len(data_x))])
X = np.asmatrix(df)
x_train, x_test, y_train, y_test = train_test_split(X, data_y, test_size=0.2,random_state=5)
y_train = np.asmatrix(y_train).transpose()
y_test = np.asmatrix(y_test).transpose()

xshape = x_train.shape[1]
weights = np.asmatrix([0 for _ in range(xshape)]).transpose() #this is wT


class LR:
    def __init__(self,weights,x_train,y_train,x_test,y_test):
        self.epochs = 10000
        self.alpha = 10e-4
        self.weights = weights
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        
    def sigmoid(self):
        product = np.dot(self.x_train,self.weights)
        return 1/(1+np.exp(-product))
    
    def gradientDescent(self):
        for i in range(self.epochs):
            xT = self.x_train.transpose()
            sigma = self.sigmoid()
            deltaY = sigma - self.y_train
            gradient = np.dot(xT,deltaY)
            self.weights = self.weights - (self.alpha * gradient)
    
        
    def prediction(self,data):
        product = np.dot(data,self.weights)
        sigma = 1/(1+np.exp(-product))
        predictions = []
        for k in range(sigma.shape[0]):
            if sigma[k] >= .5:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions = np.asmatrix(predictions).transpose()
        return predictions
    
    def error(self,x,y):
        p = self.prediction(x)
        errors = []
        for i in range(len(y)):
            errors.append((y[i] - p[i]) ** 2)

        print('Sum of Squared Errors Logistic Regression with (Batch) Gradient Descent:', sum(errors))
        if x.shape[0] < 200:
            print('Total Elements in Testing Data:', y.shape[0])
        else:
            print('Total Elements in Training Data:', y.shape[0])
        print('RMSE:', np.sqrt(sum(errors) / len(errors)))

        size = x.shape[0]
        print('Accuracy: ' + str(((size-sum(errors))/size)*100)+'%')  
    
    def convertMatrix(self,M):
        arr = []
        for i in range(M.shape[0]):
            row = []
            for j in range(M.shape[1]):
                ele = x_train[i].item(j)
                row.append(ele)
            arr.append(row)
        return arr
    
    def neuralNetwork(self):
        Y = [self.y_train[i].item(0) for i in range(self.y_train.shape[0])]
        X = self.convertMatrix(self.x_train)
        model = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(9, 2), random_state=1)
        model.fit(X,Y)
        return model
    
    def NNerror(self):
        model = self.neuralNetwork()
        p = model.predict(x_test)
        errors = []
        for i in range(len(self.y_test)):
            errors.append((self.y_test[i] - p[i]) ** 2)
            
        print('Sum of Squared Errors Neural Network with Stochastic Gradient Descent:', sum(errors))
        print('Total Elements in Testing Data:', self.y_test.shape[0])
        print('RMSE:', np.sqrt(sum(errors) / len(errors)))

        print('Accuracy: ' + str(((154-sum(errors))/154)*100)+'%')  
        #return ((154-sum(errors))/154)*100
        
    def NNerrorTrain(self):
        model = self.neuralNetwork()
        p = model.predict(self.x_train)
        errors = []
        for i in range(len(self.y_train)):
            errors.append((self.y_train[i] - p[i]) ** 2)
            
        print('Sum of Squared Errors Neural Network with Stochastic Gradient Descent:', sum(errors))
        print('Total Elements in Training Data:', self.y_train.shape[0])
        print('RMSE:',np.sqrt(sum(errors) / len(errors)))
        
        print('Accuracy: ' + str(((614-sum(errors))/614)*100)+'%')
        

lr = LR(weights,x_train,y_train,x_test,y_test)
lr.gradientDescent()
lr.error(lr.x_train,lr.y_train)
print()
lr.error(lr.x_test,lr.y_test)
print()
lr.NNerrorTrain()
print()
lr.NNerror()
