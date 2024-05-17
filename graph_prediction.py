#!/usr/bin/env python
# coding: utf-8

# In[34]:


from scipy.stats import pearsonr
import numpy as np
import pandas as pd

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [10.1, 8.4, 6.3, 5.6, 2.9]


# In[35]:


print(pearsonr(x, y)[0])

