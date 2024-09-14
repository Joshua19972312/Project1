#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv('housing.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df["ocean_proximity"].value_counts()


# In[6]:


df.describe()


# In[7]:


df.hist(bins=50, figsize=(20,15))


# In[8]:


train_set, test_set = train_test_split(df, test_size=0.2, random_state=42,stratify=None)


# In[9]:


# Bin values into discrete intervals.
df["income_cat"] = pd.cut(df["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5]) 
#cut the median income across bins and give labels to each bin


# In[10]:


df["income_cat"].hist()


# In[11]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Provides train/test indices to split data in train/test sets.
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# In[12]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[13]:


strat_train_set["income_cat"].value_counts() / len(strat_train_set)


# In[14]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[15]:


housing = strat_train_set.copy()


# In[16]:


housing.plot(kind='scatter',x='longitude',y='latitude')


# In[17]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[18]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, 
             label="population", figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()


# In[19]:


corr_matrix = housing.corr(numeric_only = True)
corr_matrix


# In[20]:


sns.heatmap(corr_matrix,annot=True)
# plt.matshow(corr_matrix)


# In[21]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[22]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)


# In[23]:


housing


# In[24]:


housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedroom_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[25]:


housing


# In[26]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[27]:


housing = strat_train_set.drop("median_house_value", axis=1)     
housing_labels = strat_train_set["median_house_value"].copy()   


# In[28]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


# In[29]:


housing_num = housing.drop("ocean_proximity", axis=1)


# In[30]:


imputer.fit(housing_num)


# In[31]:


imputer.statistics_


# In[32]:


housing_num.median().values


# In[33]:


X = imputer.transform(housing_num)         
# X = plain NumPy array containing the transformed features
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# In[34]:


housing_cat = housing[["ocean_proximity"]]


# In[35]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[36]:


housing_cat_1hot.toarray()


# In[37]:


from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[38]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),      # Imputing Missing values
    ('attribs_adder', CombinedAttributesAdder()),       # Adding attribs
    ('std_scaler', StandardScaler()),                    # Feature Scaling with Standard Scaler
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[39]:


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)             # columns with numerical attributes
cat_attribs = ["ocean_proximity"]           # columns with categorical attributes
full_pipeline = ColumnTransformer([
      ("num", num_pipeline, num_attribs),        
      ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)


# In[40]:


housing_prepared.shape


# In[41]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)


# In[42]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))


# In[43]:


from sklearn.metrics import mean_squared_error
housing_predictions=lin_reg.predict(housing_prepared)                 # predicting on training data
lin_mse = mean_squared_error(housing_labels, housing_predictions)     # calculate mean squared error
lin_rmse = np.sqrt(lin_mse)                                           # calculate root of mse
print("RMSE:",lin_rmse)
print("R-Squared:",lin_reg.score(housing_prepared,housing_labels)) 


# In[44]:


housing_labels.describe()


# In[45]:


from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)


# In[46]:


# Evaluate it on Training set
housing_predictions=tree_reg.predict(housing_prepared)
dt_mse = mean_squared_error(housing_labels, housing_predictions)     # calculate mean squared error
dt_rmse = np.sqrt(dt_mse)                                           # calculate root of mse
print("RMSE:",dt_rmse)
print("R-Squared:",tree_reg.score(housing_prepared,housing_labels))   # Return the R-squared


# In[47]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[48]:


print(tree_rmse_scores)
print("Mean:", tree_rmse_scores.mean())
print("Standard deviation:", tree_rmse_scores.std())


# In[49]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)


# In[50]:


scores=cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)


# In[51]:


print(forest_rmse_scores)
print("Mean:", forest_rmse_scores.mean())
print("Standard deviation:", forest_rmse_scores.std())


# In[52]:


from sklearn.model_selection import GridSearchCV


# In[53]:


param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[54]:


grid_search.best_params_


# In[55]:


grid_search.best_estimator_


# In[56]:


# the evaluation scores are also available
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[57]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[58]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]                   # use the categorical encoder of full pipeline
cat_one_hot_attribs = list(cat_encoder.categories_[0])                   # categorical one hot attribute
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[59]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)


# In[ ]:




