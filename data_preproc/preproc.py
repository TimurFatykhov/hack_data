
# coding: utf-8

# In[41]:


from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# In[ ]:


import pandas as pd
import numpy as np


# In[45]:


blue = pd.read_csv('blue.csv')
green = pd.read_csv('green.csv')
yellow = pd.read_csv('yellow.csv')
red = pd.read_csv('red.csv')
white = pd.read_csv('white.csv')
orange = pd.read_csv('orange.csv')
rubbish = pd.read_csv('rubbish.csv')

# посмотим на данные и увидим, насколько они "грязные"
blue.head()


# In[46]:


# соедими наши данные вместе в один большой DataFrame
df = pd.concat([blue, green, yellow, red, white, orange, rubbish], axis=0)

# посмотрим на "фигуру" нашего датасета
df.shape


# In[ ]:


# определим функцию, которая будет строку с RGB превращать в питоновский список список 
def getSepRGB(s):
  return s.strip('[]').replace(' ', '').split(',')


# In[49]:


# применим нашу новую функцию к столбцу RGB несколько раз
# и разделим компоненты по столбцам: R G B
df['R'] = df['RGB'].apply(lambda x: int(float(getSepRGB(x)[0])))
df['G'] = df['RGB'].apply(lambda x: int(float(getSepRGB(x)[1])))
df['B'] = df['RGB'].apply(lambda x: int(float(getSepRGB(x)[2])))

# удаляем столбец с RGB (больше он не нужен)
df.drop('RGB', axis=1, inplace=True)

# посмотрим что получилось
df.head(1)


# In[50]:


# теперь чуть-чуть преобразим cont, убрав с боков квадратные скобки
# и еще вишенкой на торте будет преобразование area к типу int

df['cont'] = df['cont'].apply(lambda x: x.strip('[]'))
df['area'] = df['area'].apply(int)

# посмотрим что получилось
df.head(1)


# In[ ]:


# напишем функцию, которая будет строку cont превращать в координаты
# и с помощью np.array.reshape() приведем координаты к виду
#  [ [242,123], 
#     [125,31] , 
#     [147,54] , 
#     ...        ]
def getCoordAsArray(s):
  return np.array(s.replace('\n', '').replace('  ', ' ').replace('  ', ' ').split(' ')).reshape(-1, 2)


# In[79]:


# проверим ее работу
# также по значениям проверим, похоже ли на правду
getCoordAsArray(df['cont'].iloc[0])

