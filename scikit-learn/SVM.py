import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#Generate sample table
X = np.sort(5 * np.random.rand(40, 1), axis=0)
Y = np.sin(X).ravel()

#Add noise to targets
Y[::5] += 3 * (0.5 - np.random.rand(8))

#Fit regression model
svr_rbf = SVR(kernel='rbf', gamma=0.1)
svr_lin = SVR(kernel='linear')
svr_poly = SVR(kernel='poly',degree=2)
y_rbf = svr_rbf.fit(X, Y).predict(X)
y_lin = svr_lin.fit(X, Y).predict(X)
y_poly = svr_poly.fit(X, Y).predict(X)

#Look at the result
plt.scatter(X, Y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', label='RBF')
plt.plot(X, y_lin, color='c', label='LINEAR')
plt.plot(X, y_poly,color='green', label='POLY')
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR')
#显示图例
plt.legend()
plt.show()

#---------------------------Predict XOR of inputs(SVC with RBF kernal)-------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#meshgrid生成两个同型矩阵，xx为行向量(每一行都是（-3,3,500）)，确定了矩阵列数；yy为列向量（每一列都是（-3,3,500）），确定了矩阵行数
#标准正态分布的3q原则：取值范围为（-3,3）
xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))

np.random.seed(0)
#randn:返回一个样本，具有标准正态分布
X = np.random.randn(300, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

#fit the model
#NuSVC:限制了支持向量的数量比例
clf = svm.NuSVC()
clf.fit(X, Y)

#plot the decision function for each datapoint on the grid
#计算每个样本与分离超平面之间的距离
#np.c_:将两列对应元素合在一起
#       例：xx=[1,2,3] yy=[4,5,6] return:[[1,4],
#                                         [2,5],
#                                         [3,6]]
#其实是构建了一个500*500的网格
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#imshow()可以显示任意的二维数组;
#将矩阵Z以image的形式显示出来，距离最近插值，坐标范围定义为extent，原点在下方，colormap=‘自定义’，
plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)

#contour:绘制等值线
#levels=[0]表示绘制0等值线，即支持向量所在的曲线。
contours = plt.contour(xx, yy, Z, levels=[0], linewidth=2, linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)

#设置x，y轴的标签,若为空则不显示坐标信息。
plt.xticks(())
plt.yticks(())
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()

#--------------------------------SVM(with linear kernal)---------------------------------
#-------------------------Maximum margin separating hyperplane---------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
np.random.seed(0)

#对应于np.c_。np.r_则按照y轴排列元素
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0]*20 + [1]*20

#fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

#get the separating hyperplane
#仅在线性核中有  
#====================================
# w[0]*x0 + w[1]*x1 + b = 0 超平面方程
# x1 = -(w[0] / w[1])*x0 - b / w[1]
#====================================
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

#plot the parallels to the separating hyperplane that pass the
#support vectors
#第一个和最后一个一定是不同类的支持向量
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx +(b[1] - a * b[0])

#plot the line, the points, and the nearst vectors to the plane
#k:black
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, -1], s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

#'tight'使坐标轴的范围和数据保持一致。
plt.axis('tight')
plt.show()

#------------------------Separating hyperplane for unblanced classes-------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#we create 40 separable points
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2),
          0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)

#fit the model and get the separating hyperplane
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]

#get the separating hyperplane using weighted classes
#对类别为1的增加权重
wclf = svm.SVC(kernel='linear', class_weight={1:10})
wclf.fit(X, y)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]

#plot separating hyperplane and samples
plt.plot(xx, yy, 'k-', label='no weights')
plt.plot(xx, wyy, 'g--', label='with weights')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.legend()
plt.axis('tight')
plt.show()

#--------------------------SVM  with univariate feature selection----------------------
import numpy as np
from sklearn import svm, datasets, feature_selection
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

digits = datasets.load_digits()
y = digits.target
y = y[:200]
X = digits.data[:200]
n_sample = len(y)
X = X.reshape((n_sample, -1))
#并列
X = np.hstack((X, 2 * np.random.random((n_sample, 200))))
#create an feature-selection transfrom and an instance of SVM that
#we combine together to have an full-blown estimator
transfrom = feature_selection.SelectPercentile(feature_selection.f_classif)