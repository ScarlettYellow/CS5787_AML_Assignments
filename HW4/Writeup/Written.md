

# Written Exercises

## Written 1

Question: Show that from the Singular Value Decomposition(SVD) of a matrix $X$, we can obtain the eigendecomposition of $X^TX$. This tells us that we can do an SVD of $X$ and get same result as the eigendecomposition of $X^TX$ but the SVD is faster and easier.

Answer:

The eigenvalue decomposition of a real symmetric matrix can be written as follows:
$$
X = Q \Lambda Q^T
$$
Assuming that matrix $X$ is an $m*n$ matrix, we can define the SVD of matrix $X$ as follows:
$$
X = U \Sigma V^T
$$
The columns of $V$ are composed of the eigenvectors of $X^TX$, and the eigenvectors are unit column vectors. 

Here's the proof:
$$
X = U \Sigma V^T
$$

$$
\Rightarrow X^T = V \Sigma^{T} U^T
$$

$$
\Rightarrow X^TX = V \Sigma^{T} U^T U \Sigma V = V \Sigma \Sigma^{T} V^T
$$

Note: $\Sigma$ is a $m*n$ matrix, except for the elements on the main diagonal, the value of other elements are all zero. Every element on the main diagonal is a singular value. So we have:
$$
\Sigma^T \Sigma = \Sigma^2
$$
Therefore, we can get:
$$
\Rightarrow X^TX = V \Sigma^2 V^T
$$
Here, $\Sigma$ is a diagonal matrix with singular values as its elements, and $\Sigma \Sigma^T $ is a diagonal matrix consisting of the square of singular values. 

Thus, we can see that as a symmetric matrix, the form of the equation (7) of $X^TX$ above is the same as the eigenvalue decomposition of a symmetric matrix.

Therefore, we can know that:

- The nonzero singular values of matrix $X$ is the positive square root of the nonzero eigenvalues of $X^TX$.
- The orthogonal matrix $V$ is the eigenvector of $X^TX$.

In conclusion, we can obtain the eigendecomposition of $X^TX$ from the Singular Value Decomposition(SVD) of a matrix $X$ very fast and easily.



## Written 2

$$
z_{il} = x_{il}(\frac{w_l}{\sum^p_{l=1}w_l})^{\frac{1}{2}} \Rightarrow z_{i^{'}l} = x_{i^{'}l}(\frac{w_l}{\sum^p_{l=1}w_l})^{\frac{1}{2}}
$$


$$
d_e^{(w)}(x_i,x_{i^{'}}) = \sum^p_{l=1}(z_{il}-z_{i^{'}l})^2
$$

$$
= \sum^p_{l=1}[x_{il}(\frac{w_l}{\sum^p_{l=1}w_l})^{\frac{1}{2}}-x_{i^{'}l}(\frac{w_l}{\sum^p_{l=1}w_l})^{\frac{1}{2}}]^2
$$

$$
= \sum^p_{l=1}[(x_{il}-x_{i^{'}l})(\frac{w_l}{\sum^p_{l=1}w_l})^{\frac{1}{2}}]^2
$$

$$
= \sum^p_{l=1}(x_{il}-x_{i^{'}l})^2(\frac{w_l}{\sum^p_{l=1}w_l})
$$

$$
=\frac{\sum^p_{l=1}w_l(x_{il}-x_{i^{'}l})^2}{\sum^p_{l=1}w_l}
$$



## Written 3

```python
# import modules

import numpy as np
import math
```


```python
# create the matrix

M = np.array([[1,0,3],[3,7,2],[2,-2,8],[0,-1,1],[5,8,7]])
M = np.mat(M)
print (M)
print (M.shape)
print (type(M))
```

    [[ 1  0  3]
     [ 3  7  2]
     [ 2 -2  8]
     [ 0 -1  1]
     [ 5  8  7]]
    (5, 3)
    <class 'numpy.matrix'>



```python
# (a) Compute the matrices M^T*M and M*M^T

MT = M.T
print (MT)
print (MT.shape)
```

    [[ 1  3  2  0  5]
     [ 0  7 -2 -1  8]
     [ 3  2  8  1  7]]
    (3, 5)



```python
# a-1: Compute the matrix M^T*M

MTM = MT*M
print (MTM)
print (MTM.shape)
```

    [[ 39  57  60]
     [ 57 118  53]
     [ 60  53 127]]
    (3, 3)



```python
# a-2: Compute the matrix M*M^T

MMT = M*MT
print (MMT)
print (MMT.shape)
```

    [[ 10   9  26   3  26]
     [  9  62   8  -5  85]
     [ 26   8  72  10  50]
     [  3  -5  10   2  -1]
     [ 26  85  50  -1 138]]
    (5, 5)



```python
# (b)&(c) 

# 1.Find the eigenvalues and eigenvectors for M^T*M

MTM_E, MTM_V = np.linalg.eig(MTM)

print ("eigenvalues=", MTM_E)
print (type(MTM_E))
print ("eigenvector=", MTM_V)
print (type(MTM_V))
```

    eigenvalues= [2.14670489e+02 9.32587341e-15 6.93295108e+01]
    <class 'numpy.ndarray'>
    eigenvector= [[ 0.42615127  0.90453403 -0.01460404]
     [ 0.61500884 -0.30151134 -0.72859799]
     [ 0.66344497 -0.30151134  0.68478587]]
    <class 'numpy.matrix'>



```python
# 1.Find the eigenvalues and eigenvectors for M*M^T

MMT_E, MMT_V = np.linalg.eig(MMT)

print ("eigenvalues=", MMT_E)
print (type(MMT_E))
print ("eigenvector=", MMT_V)
print (type(MMT_V))
```

    eigenvalues= [ 2.14670489e+02 -8.88178420e-16  6.93295108e+01 -3.34838281e-15
      7.47833227e-16]
    <class 'numpy.ndarray'>
    eigenvector= [[-0.16492942 -0.95539856  0.24497323 -0.54001979 -0.78501713]
     [-0.47164732 -0.03481209 -0.45330644 -0.62022234  0.30294097]
     [-0.33647055  0.27076072  0.82943965 -0.12704172  0.2856551 ]
     [-0.00330585  0.04409532  0.16974659  0.16015949  0.43709105]
     [-0.79820031  0.10366268 -0.13310656  0.53095405 -0.13902319]]
    <class 'numpy.matrix'>



```python
# (d) Method 1

# Find the SVD for the original matrix M from parts (b) and (c)

# d-1: Find the sigma of matrix M
# The result is the same when replacing MMT_E with MTM_E, because MTM and MMT have same non-zero eigenvalues, index=0,2  
M_sigma = np.array([math.sqrt(MMT_E[0]),math.sqrt(MMT_E[2])])
print ("sigma=", M_sigma) 

# check the number of non-zero eigenvalues, = the rank of matrix 
M_rank = np.linalg.matrix_rank(M)
print (M_rank) # M is rank 2.
```

    sigma= [14.65163776  8.32643446]
    2



```python
# d-2: Find the VT of matrix M, VT is a 2*2 matrix 

rows_V = [0,1] # get row 0 and 1
cols_V = [0,2] # get column 0 and 2
M_V = MTM_V[rows_V,:][:,cols_V]
M_VT = M_V.T
print ("VT=", M_VT)
```

    VT= [[ 0.42615127  0.61500884]
     [-0.01460404 -0.72859799]]



```python
# d-3: Find the U of matrix M, U is a 5*2 matrix 

cols_U = [0,2] #  get column 0 and 2
M_U = MMT_V[:,cols_U]
print ("U=", M_U)
```

    U= [[-0.16492942  0.24497323]
     [-0.47164732 -0.45330644]
     [-0.33647055  0.82943965]
     [-0.00330585  0.16974659]
     [-0.79820031 -0.13310656]]



```python
# (e) Method 1

# e-1: keep only one non-zero singular value, by setting the smaller singular value to 0

sigma_max = max(M_sigma)
print (sigma_max)
M_sigma_new = np.array([sigma_max]) 
#M_sigma_new = np.mat(M_sigma_new)
print (M_sigma_new)
```

    14.651637764976883
    [14.65163776]



```python
# e-2: Compute the 1D approximation to M

k=1
u,d,vt = M_U[:,:k],M_sigma[:k],M_VT[:,:k][:k,:]
print("------U-----")
print(u)
print("------S-----")
print(d)
print("------VT-----")
print(vt)

# Compute the 1D approximation to M

A = np.zeros([1,1])
for i in range(1):
    A[i][i] = d[i]
print (A)
tmp = np.dot(u,A)
print("1D approximation to M:")
print(np.dot(tmp,vt))
```

    ------U-----
    [[-0.16492942]
     [-0.47164732]
     [-0.33647055]
     [-0.00330585]
     [-0.79820031]]
    ------S-----
    [14.65163776]
    ------VT-----
    [[0.42615127]]
    [[14.65163776]]
    1D approximation to M:
    [[-1.02978864]
     [-2.94487812]
     [-2.10085952]
     [-0.02064112]
     [-4.9838143 ]]



```python
# (d) Method 2

U,sigma,VT = np.linalg.svd(M)

print ("U=", U)
print ("sigma=", sigma)
print ("VT=", VT)

k=2
u,d,vt = U[:,:k],sigma[:k],VT[:,:k][:k,:]
print("------U-----")
print(u)
print("------S-----")
print(d)
print("------VT-----")
print(vt)
```

    U= [[-0.16492942 -0.24497323  0.9482579   0.09864471 -0.06214956]
     [-0.47164732  0.45330644 -0.02261948  0.08103373 -0.75165416]
     [-0.33647055 -0.82943965 -0.27341434 -0.18350729 -0.3006445 ]
     [-0.00330585 -0.16974659 -0.14522096  0.97468061  0.00915155]
     [-0.79820031  0.13310656 -0.06671416  0.00505374  0.58368021]]
    sigma= [1.46516378e+01 8.32643446e+00 2.99921582e-16]
    VT= [[-0.42615127 -0.61500884 -0.66344497]
     [ 0.01460404  0.72859799 -0.68478587]
     [-0.90453403  0.30151134  0.30151134]]
    ------U-----
    [[-0.16492942 -0.24497323]
     [-0.47164732  0.45330644]
     [-0.33647055 -0.82943965]
     [-0.00330585 -0.16974659]
     [-0.79820031  0.13310656]]
    ------S-----
    [14.65163776  8.32643446]
    ------VT-----
    [[-0.42615127 -0.61500884]
     [ 0.01460404  0.72859799]]



```python
# (e) Method 2

k=1
u,d,vt = u[:,:k],d[:k],vt[:,:k][:k,:]
print("------U-----")
print(u)
print("------S-----")
print(d)
print("------VT-----")
print(vt)

# Compute the 1D approximation to M
A = np.zeros([1,1])
for i in range(1):
    A[i][i] = d[i]
tmp = np.dot(u,A)
print("1D approximation to M:")
print(np.dot(tmp,vt))
```

    ------U-----
    [[-0.16492942]
     [-0.47164732]
     [-0.33647055]
     [-0.00330585]
     [-0.79820031]]
    ------S-----
    [14.65163776]
    ------VT-----
    [[-0.42615127]]
    1D approximation to M:
    [[1.02978864]
     [2.94487812]
     [2.10085952]
     [0.02064112]
     [4.9838143 ]]



















































































































































