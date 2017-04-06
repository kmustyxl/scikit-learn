#----------------------------------稀疏约束Lasso-------------------------------

from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([1, 1])

#---------------------------------最小角回归 lars-------------------------------

from sklearn import linear_model
reg = linear_model.LassoLars(alpha=.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.coef_

'''
Computing regularization path using the LARS...

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()
#442个样本，10个特征
X = diabetes.data
#442个标签
y = diabetes.target
print('Computing regularization path using the LARS...')

#coefs=得到13组系数，但默认输出是（coefs.T 10*13）
alphas, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)

#得到13组系数的绝对值加权和
xx = np.sum(np.abs(coefs.T), axis = 1)

#归一化处理：因为越往后，对系数的约束越小，则系数越大。
xx /= xx[-1]

plt.plot(xx, coefs.T)

#得到y轴的最小值和最大值
ymin, ymax = plt.ylim()

#以xx为单位，画垂直于x轴的虚线（dashed），每条曲线始于ymin，终于ymax
plt.vlines(xx, ymin, ymax, linestyle='dashed')

plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.show()

#-------------------------------------Compressive sensing压缩感知-------------------------------
import numpy as np
from scipy import sparse
from scipy import ndimage
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

def _weights(x, dx=1, orig=0):
    #ravel:返回扁平化后的一维数组
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx)
    alpha = (x - orig - floor_x * dx) / dx
    #hstack:堆栈以列进行排序
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))

def _generate_center_coordinates(l_x):
    #mgrid：分割坐标轴
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y

def build_projection_operator(l_x, n_dir):
    '''
    Comupte the tomography design matrix.

    Parameters
    ----------
    l_x : int
          linear size of image array

    n_dir : int
          number of angles at which projections are required.

    Returns
    -------
    p : sparse matrix of shape (n_dir l_x, l_x**2)
    '''
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x**2)
    data_unravel_indices = np.hstack((data_unravel_indices, data_unravel_indices))

    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds <= l_x)
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    return proj_operator

def generate_synthetic_data():
    '''
    Synthetic binary data
    '''
    #设定一个随机状态，使得随机数可以预测，而不是每次都是random生成。
    rs = np.random.RandomState(0)
    n_pts = 36.
    x, y = np.ogrid[0:1, 0:1]
    mask_outer = (x - 1 / 2) ** 2 + (y - 1 / 2) ** 2 < (1 / 2) ** 2
    mask = np.zeros((1, 1))
    points = 1 * rs.rand(2, n_pts)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigmal=1 / n_pts)
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return res - ndimage.binary_erosion(res)

#Generate synthetic images, and projections
l = 128
proj_operator = build_projection_operator(1, 1 / 7.)
data = generate_synthetic_data()
proj = proj_operator * data.ravel()[:, np.newaxis]
proj += 0.15 * np.random.randn(*proj.shape)

# Reconstruction with L2(Ridge) penalization
rgr_ridge = Ridge(alpha=0.2)
rgr_radge.fit(proj_operator, proj.ravel())
rec_l2 = rgr_ridge.coef_.reshape(1, 1)

# Reconstruction with L1(Lasso) penalization
# the best value of alpha was determined using cross validation
# with LassoCV
rgr_lasso = Lasso(alpha=0.001)
rgr_lasso.fit(proj_operator, proj.ravel())
rec_l1 = rgr_lasso.coef_.reshape(1, 1)

plt.figure(figsize=(8, 3.3))
plt.subplot(131)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(rec_l2, cmap=plt.cm.gray, interpolation='nearest')
plt.title('L2 penalization')
plt.axis('off')
plt.subplot(133)
plt.imshow(rec_l1, cmap=plt.cm.gray, interpolation='nearest')
plt.title('L1 penalization')
plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
plt.show()