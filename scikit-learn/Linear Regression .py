'''
Supervised learning

Data:2017-04-01
'''
#------------------------------------Liner Regression code-------------------------------------------
#调用class：线性模型
from sklearn import linear_model
#调用该类中的线性回归mothed
reg = linear_model.LinearRegression()
#拟合X,Y。
#Y = w0 + w1x1 + w2x2 + ... + wpxp
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
#vector w = (w1, w2,...wp)
reg.coef_
#w0
reg.intercept_
print('qweqweqwe')
#------------------------------------Liner Regression Example-----------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

#Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

#Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#Split the targets into training/testing sets
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

#Create linear regression object
regr = linear_model.LinearRegression()

#Train the model using the training set
regr.fit(diabetes_X_test, diabetes_Y_test)

#The coefficients
print('Coefficients: \n', regr.coef_)
#The mean squared error
print('Mean squared error: %.2f' %np.mean((regr.predict(diabetes_X_test) - diabetes_Y_test) ** 2))
#Explained variance score
print('Variance score: %.2f' %regr.score(diabetes_X_test, diabetes_Y_test))

#Plot outputs
plt.scatter(diabetes_X_test, diabetes_Y_test, color = 'black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()



