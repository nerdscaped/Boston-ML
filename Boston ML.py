#Boston Machine Learning
#Code by Matt Cadel
#Started 26/03/22

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import scipy as sp
from scipy.stats import chi2
from sklearn.metrics import r2_score
from scipy.stats.mstats import winsorize
from sklearn import metrics
from sklearn.experimental import enable_iterative_imputer  
from sklearn.covariance import MinCovDet
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline

#Boston Data Prep
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['PRICE'] = boston.target

#EDA to undestand the data
print(boston.DESCR)
print(boston_df.head(5))
print(boston_df.columns)
print(boston_df.isnull().sum()) #Check for nulls - no nulls present
correlation_matrix = boston_df.corr()
sns.heatmap(data=correlation_matrix, annot=True) #Looking good, but this is not enough. We should also have a visual on the correlations to look out for outliers

variables = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

plt.figure(figsize=(25,40))
for i, col in enumerate(variables):
    ax = plt.subplot(3, 5, i+1)
    boston_df.plot.scatter(x=col, y='PRICE', ax=ax, label=col, legend=False,s=2)
    ax.set_title(col)
plt.tight_layout()
#plt.show() #This is highly informative, however a measure of the extent of outliers is needed for a more scientific analysis

#I will be using the Tukey method to identify the number of outliers of each variable

def tukeys_method(df, variable): #This function comes from Alicia Horsch - https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755 Else this is all my code
    #Takes two parameters: dataframe & variable of interest as string
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3-q1
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr

    #inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence
    
    #outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    
    outliers_prob = []
    outliers_poss = []
    for index, x in enumerate(df[variable]):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
    for index, x in enumerate(df[variable]):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)
    return outliers_prob, outliers_poss

for i, col in enumerate(variables):
    proportion_outliers = round((len(tukeys_method(boston_df,col)[1]) / len(boston_df[col]))*100,2)
    print("Percentage of",col,"observations that are outliers:\n",proportion_outliers,"%")

#Reported outlier percentages vary from 0 to 100% (100% from dummy variable). 
#Outliers come in two forms, those that are actual cases of unusual data (non-error outliers), and those that come from wrongful data collection (error outliers)
#Very useful - I will fit the model with and without the B variable to see how things change, as it currently has the highest number of outliers, and likely is correlated with other variables, namely crime and income

#Multivariate Outliers - these are observations that have unusual figures over multiple variables
#I'm using Robust Mahalanobis Distance to identify these cases, again on the advice of Alicia Horsch

def robust_mahalanobis_method(df):
    #Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df.values.T)
    X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_ #robust covariance metric
    robust_mean = cov.location_  #robust mean
    inv_covmat = sp.linalg.inv(mcd) #inverse covariance metric
    
    #Robust M-Distance
    x_minus_mu = df - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())
    
    #Flag as outlier
    outlier = []
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))#degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier

print(robust_mahalanobis_method(boston_df),len(robust_mahalanobis_method(boston_df))) #These are the indexes of the multivariate outliers
#27 multivariate outliers have been identified. Again, I will run the ML with and without these data points

#--------------------------------Machine Learning - 1 - raw Boston dataset--------------------------------
x = boston_df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = boston_df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=69) 
lm = LinearRegression()
lm.fit(X_train,y_train)

#Testing Model
y_train_predict = lm.predict(X_train)
y_test_predict = lm.predict(X_test)

#Training Set
print("Training Set")
print('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
print('R SQUARED', r2_score(y_train, y_train_predict))

#Testing Set
print("Testing Set")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
print('R SQUARED', r2_score(y_test, y_test_predict))

#--------------------------------Machine Learning - 2 - without B variable--------------------------------
x = boston_df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']]
y = boston_df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=69) 
lm = LinearRegression()
lm.fit(X_train,y_train)

#Testing Model
y_train_predict = lm.predict(X_train)
y_test_predict = lm.predict(X_test)

#Training Set
print("Training Set")
print('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
print('R SQUARED', r2_score(y_train, y_train_predict))

#Testing Set
print("Testing Set")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
print('R SQUARED', r2_score(y_test, y_test_predict))
#Very similiar values to before, however are slightly declined. 

#--------------------------------Machine Learning - 3 - without B variable outliers--------------------------------
boston_df_no_B_OL = boston_df.drop(tukeys_method(boston_df,'B')[0])  #Only the probable outliers removed

x = boston_df_no_B_OL[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = boston_df_no_B_OL['PRICE']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=12) 
lm = LinearRegression()
lm.fit(X_train,y_train)

#Testing Model
y_train_predict = lm.predict(X_train)
y_test_predict = lm.predict(X_test)

#Training Set
print("Training Set")
print('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
print('R SQUARED', r2_score(y_train, y_train_predict))

#Testing Set
print("Testing Set")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
print('R SQUARED', r2_score(y_test, y_test_predict))
#Better fit on the training set, but worse on the testing

#--------------------------------Machine Learning - 4 - without ZN variable outliers--------------------------------
boston_df_no_ZN_OL = boston_df.drop(tukeys_method(boston_df,'ZN')[0])  #Only the probable outliers removed

x = boston_df_no_ZN_OL[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = boston_df_no_ZN_OL['PRICE']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=12) 
lm = LinearRegression()
lm.fit(X_train,y_train)

#Testing Model
y_train_predict = lm.predict(X_train)
y_test_predict = lm.predict(X_test)

#Training Set
print("Training Set")
print('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
print('R SQUARED', r2_score(y_train, y_train_predict))

#Testing Set
print("Testing Set")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
print('R SQUARED', r2_score(y_test, y_test_predict))
#Better fit on the training set, but worse on the testing

#--------------------------------Machine Learning - 5 - without multivariate outliers--------------------------------
boston_df_no_MV = boston_df.drop(robust_mahalanobis_method(boston_df))

x = boston_df_no_MV[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = boston_df_no_MV['PRICE']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12) 
lm = LinearRegression()
lm.fit(X_train,y_train)

#Testing Model
y_train_predict = lm.predict(X_train)
y_test_predict = lm.predict(X_test)

#Training Set
print("Training Set")
print('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
print('R SQUARED', r2_score(y_train, y_train_predict))

#Testing Set
print("Testing Set")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
print('R SQUARED', r2_score(y_test, y_test_predict)) #Looking better :) 