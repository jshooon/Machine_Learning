#!/usr/bin/env python
# coding: utf-8

# ### 머신러닝 필요 라이브러리 설치

# In[31]:


import sys

# 사이킷런 최신 버전을 설치합니다.
get_ipython().system('pip install -q --upgrade scikit-learn')
# mglearn을 다운받고 압축을 풉니다.
get_ipython().system('wget -q -O mglearn.tar.gz https://bit.ly/mglearn-tar-gz')
get_ipython().system('tar -xzf mglearn.tar.gz')


# In[33]:


import sklearn
from preamble import *


# In[26]:


get_ipython().system('pip install numpy scipy matplotlib ipython scikit-learn pandas pillow imageio')


# ### Numpy 배열의 예

# In[8]:


import numpy as np

x = np.array([[1,2,3], [4,5,6]])
print('x:\n',x)


# In[9]:


from scipy import sparse

# 대각선 원소는 1이고 나머지는 0인 2차원 NumPy 배열을 만든다.
eye = np.eye(4)
print('NumPy 배열:\n', eye)


# In[11]:


# NumPy 배열을 CSR 포맷의 SciPy 희박 행렬로 변환한다.
# 0이 아닌 원소만 저장된다.
sparse_matrix = sparse.csr_matrix(eye)
print('\nSciPy의 CSR 행렬:\n', sparse_matrix)


# In[12]:


data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print('COO 표현:\n', eye_coo)


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# -10에서 10까지 100개의 간격으로 나뉘어진 배열을 생성합니다.
x = np.linspace(-10, 10, 100)
# 사인 함수를 사용하여 y 배열을 생성합니다.
y = np.sin(x)
# plot 함수는 한 배열의 값을 다른 배열에 대응해서 선 그래프를 그립니다.
plt.plot(x, y, marker='x')


# In[17]:


import pandas as pd

# 회원 정보가 들어간 간단한 데이터셋을 생성합니다.
data = {'Name' : ['Jogn', 'Anna', 'Peter', 'Linda'],
           'Location' : ['New York', 'Paris', 'Berlin', 'London'],
           'Age' : [24, 13, 53, 33]
           }

data_pandas = pd.DataFrame(data)
# 주피터 노트북은 DataFrame을 미려하게 출력해줍니다.
data_pandas


# In[18]:


# Age 열의 값이 30 이산인 모든 행을 선택합니다.
data_pandas[data_pandas.Age > 30]


# In[21]:


pip install mglearn


# In[70]:


import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn


# In[71]:


import sys
print("Python 버전:", sys.version)

import pandas as pd
print("pandas 버전:", pd.__version__)

import matplotlib
print("matplotlib 버전:", matplotlib.__version__)

import numpy as np
print("NumPy 버전:", np.__version__)

import scipy as sp
print("SciPy 버전:", sp.__version__)

import IPython
print("IPython 버전:", IPython.__version__)

import sklearn
print("scikit-learn 버전:", sklearn.__version__)


# ### 1.7 첫 번째 애플리케이션 : 붓꽃의 품종 분류
# * 1.7.1 데이터 적재

# In[72]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[73]:


print("iris_dataset의 키:\n", iris_dataset.keys())


# In[74]:


print(iris_dataset['DESCR'][:193] + "\n...")


# In[75]:


print("타깃의 이름:", iris_dataset['target_names'])


# In[76]:


print("특성의 이름:\n", iris_dataset['feature_names'])


# In[77]:


print("data의 타입:", type(iris_dataset['data']))


# In[78]:


print("data의 크기:", iris_dataset['data'].shape)


# In[79]:


print("data의 처음 다섯 행:\n", iris_dataset['data'][:5])


# In[80]:


print("target의 타입:", type(iris_dataset['target']))


# In[81]:


print("target의 크기:", iris_dataset['target'].shape)


# In[82]:


print("타깃:\n", iris_dataset['target'])


# In[83]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[84]:


print("X_train 크기:", X_train.shape)
print("y_train 크기:", y_train.shape)


# In[86]:


print("X_test 크기:", X_test.shape)
print("y_test 크기:", y_test.shape)


# In[87]:


# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용합니다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show() # 책에는 없음


# In[88]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[89]:


knn.fit(X_train, y_train)


# In[90]:


X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)


# In[91]:


prediction = knn.predict(X_new)
print("예측:", prediction)
print("예측한 타깃의 이름:", 
       iris_dataset['target_names'][prediction])


# In[92]:


y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n", y_pred)


# In[93]:


print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))


# In[67]:


print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))


# In[94]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))

