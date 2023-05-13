#!/usr/bin/env python
# coding: utf-8

# In[88]:


pip install github


# In[89]:


import pandas as pd


# In[90]:


df1 = pd.read_csv("D:\DL Drew\Linear Regression Datasets\houseprice.csv")


# In[91]:


df1.head()


# In[92]:


df1.tail()


# In[93]:


df1.info()


# In[94]:


df1.describe()


# In[95]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[96]:


fig = sns.displot(df1,x = "bedrooms")
fig


# In[97]:


fig = sns.displot(df1,x = "sqft_lot")
fig


# In[98]:


fig = sns.displot(df1,x = "date")


# In[99]:


fig = px.histogram(df1,x = "bedrooms",color_discrete_sequence = ["teal"])
fig.update_layout(bargap=0.2)
fig.show()


# In[100]:


df1.bedrooms.describe()


# In[101]:


fig = px.histogram(df1, 
                   x='bedrooms', 
                   marginal='box', 
                   nbins=47, 
                   title='Bedrooms check')
fig.update_layout(bargap=0.2)
fig.show()


# In[102]:


fig = px.histogram(df1, 
                   x='bathrooms', 
                   marginal='box', 
                   color_discrete_sequence=['red'], 
                   title='Bathrooms check')
fig.update_layout(bargap=0.1)
fig.show()


# In[103]:


df1.head()


# In[104]:


fig = px.histogram(df1, 
                   x='price', 
                   marginal='box', 
                   color='floors', 
                   color_discrete_sequence=['red', 'blue'], 
                   title='Total House Price')
fig.update_layout(bargap=0)
fig.show()


# In[105]:


df1.floors.corr(df1.price)


# In[106]:


df1.floors


# In[107]:


df1.info()


# In[108]:


df1.sqft_living.corr(df1.price)


# In[109]:


df1.sqft_living


# In[110]:


df1.sqft_living.value_counts()


# In[111]:


df1.corr()


# In[112]:


sns.heatmap(df1.corr(), cmap='Reds', annot=True)
plt.title('Correlation')


# In[117]:


plt.title('Sqft Living vs. Price')
sns.scatterplot(data=df1, x='sqft_living', y='price', alpha=0.7, s=15);


# In[121]:


def predict_price(sqft_living, w, b):
    return w * sqft_living + b


# In[122]:


w = 50
b = 100


# In[123]:


sqft = df1.sqft_living
predicted_price = predict_price(sqft, w, b)


# In[124]:


plt.plot(sqft, predicted_price, 'r-o');
plt.xlabel('Sqft');
plt.ylabel('Predicted House Price');


# In[127]:


target = df1.price

plt.plot(sqft, predicted_price, 'r', alpha=0.9);
plt.scatter(sqft, target, s=8,alpha=0.8);
plt.xlabel('Sqft');
plt.ylabel('Price')
plt.legend(['Predict', 'Actual']);


# In[ ]:




