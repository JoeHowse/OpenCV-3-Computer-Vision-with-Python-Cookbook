#!/usr/bin/env python
# coding: utf-8

# In[24]:


import cv2
import numpy as np


# In[65]:


N = 10
A = np.random.randn(N,N)
while np.linalg.matrix_rank(A) < N:
    A = np.random.randn(N,N)
x = np.random.randn(N,1)
b = A @ x


# In[69]:


ok, x_est = cv2.solve(A, b)
print('Solved:', ok)
if ok:
    print('Residual:', cv2.norm(b - A @ x_est))
    print('Relative error:', cv2.norm(x_est - x) / cv2.norm(x))


# In[70]:


N = 10
A = np.random.randn(N*2,N)
while np.linalg.matrix_rank(A) < N:
    A = np.random.randn(N*2,N)
x = np.random.randn(N,1)
b = A @ x


# In[71]:


ok, x_est = cv2.solve(A, b, flags=cv2.DECOMP_NORMAL)
print('Solved overdetermined system:', ok)
if ok:
    print('Residual:', cv2.norm(b - A @ x_est))
    print('Relative error:', cv2.norm(x_est - x) / cv2.norm(x))


# In[72]:


N = 10
A = np.random.randn(N,N*2)
x = np.random.randn(N*2,1)
b = A @ x


# In[73]:


w, u, v_t = cv2.SVDecomp(A, flags=cv2.SVD_FULL_UV)
mask = w > 1e-6
w[mask] = 1 / w[mask]
w_pinv = np.zeros((A.shape[1], A.shape[0]))
w_pinv[:N,:N] = np.diag(w[:,0])
A_pinv = v_t.T @ w_pinv @ u.T
x_est = A_pinv @ b


# In[74]:


print('Solved underdetermined system')
print('Residual:', cv2.norm(b - A @ x_est))
print('Relative error:', cv2.norm(x_est - x) / cv2.norm(x))


# In[ ]:




