#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


df1 = pd.read_csv('train_values.csv')
df2 = pd.read_csv('train_labels.csv')


# In[45]:


df3 = pd.merge(df1, df2, on='building_id', how='inner')


# In[46]:


df3.isnull().sum()


# In[47]:


df3.info()


# In[48]:


no = df3.groupby('damage_grade')['building_id'].count().sort_values(ascending = False).reset_index()
no.columns = ['damage_grade', 'count']
sns.barplot(data = no,x = 'damage_grade',y = 'count')


# In[49]:


df4 = df3.select_dtypes(include='object')
df4.columns


# In[50]:


col = ['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'legal_ownership_status']


# In[51]:


df3.drop(columns = col,axis = 1,inplace = True)


# In[52]:


df3.columns


# In[53]:


damaged_buildings = df3[df3['damage_grade']==3]
damaged_buildings


# In[54]:


secondary_use_columns = [
    'has_secondary_use_hotel', 
    'has_secondary_use_rental',
    'has_secondary_use_institution', 
    'has_secondary_use_school',
    'has_secondary_use_industry', 
    'has_secondary_use_health_post',
    'has_secondary_use_gov_office', 
    'has_secondary_use_use_police']

# sec_col = [i.split('_')[3:5] for i in secondary_use_columns ]
# sec_col_join = ['_'.join(i) for i in sec_col ]
# sec_col_join
# df3.rename(columns = dict(zip(secondary_use_columns, sec_col_join)), inplace=True)


# In[13]:


damaged_counts = damaged_buildings[secondary_use_columns].sum().reset_index()
damaged_counts.columns = ['building','count']
sort = damaged_counts.sort_values(by = 'count',ascending = False)
ax =sns.barplot(data = sort,x = 'building' , y = 'count')
for i in ax.containers:
    ax.bar_label(i)
    
plt.xticks(rotation=90)  
plt.show()
    


# In[58]:


df3.columns


# In[59]:


structures = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
       'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other']


# In[60]:


damaged_structures = df3[df3['damage_grade']==3]
damaged_structures_count = damaged_structures[structures].sum().reset_index()
damaged_structures_count.columns = ['structures','count']
sort = damaged_structures_count.sort_values(by = 'count',ascending = False)
dx = sns.barplot(data = sort,x = 'structures' , y = 'count' )
for i in dx.containers:
    dx.bar_label(i)
    
plt.xticks(rotation=90)  
plt.show()


# In[61]:


correlation = df3[['age', 'damage_grade']].corr()
correlation


# In[62]:


sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap: Age vs Damage Grade")
plt.show()


# In[63]:


import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'df3' is your DataFrame and 'damage_grade' is the target variable
# Shift the labels of 'damage_grade' to start from 0
df3['damage_grade'] = df3['damage_grade'] - 1
# Separate the features (X) and target (y)
X = df3.drop('damage_grade', axis=1)  # Drop the target column from the features
y = df3['damage_grade']  # Target variable

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




# In[64]:


df5 = pd.read_csv('test_values.csv')


# In[65]:


col = ['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'legal_ownership_status']
df5.drop(columns = col,axis = 1,inplace = True)


# In[66]:


df5.columns


# In[75]:


y_pred = model.predict(df5)


# In[77]:


output = pd.DataFrame({'building_id':df5["building_id"],'predicted_values':y_pred})


# In[79]:


output['predicted_values'] = output['predicted_values'] +1


# In[81]:


output.to_csv('submission_format.csv',index = False)


# In[ ]:




