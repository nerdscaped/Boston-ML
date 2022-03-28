# Boston ML

The aim of this project was to predict the value of Bostonian homes as accurately as possible using the scikit-learn library in python. 

The variables I used to do this include:
- CRIM     per capita crime rate by town
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    proportion of non-retail business acres per town
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      nitric oxides concentration (parts per 10 million)
- RM       average number of rooms per dwelling
- AGE      proportion of owner-occupied units built prior to 1940
- DIS      weighted distances to five Boston employment centres
- RAD      index of accessibility to radial highways
- TAX      full-value property-tax rate per $10,000
- PTRATIO  pupil-teacher ratio by town
- B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
- LSTAT    % lower status of the population
- PRICE     Median value of owner-occupied homes in $1000's

I went through 5 iterations of ML, and eventually landed on a model with a testing R^2 of 0.814, an impressive improvement from 0.745 of the raw dataset.

### The Process

##### Loading Packages
The first step was to load in the packages needed for the project. These included Pandas, NumPy, Matplotlib, Seaborn, Sci-kit learn and SciPy.

##### Loading The Dataset
The Boston dataset is included in the Sci-kit learn dataset. It was then put into a Pandas array for analysis.

##### Exploratory Data Analysis
After this, I loaded in the dataset and performed some preliminary exploratory data analysis. For example, I checked whether there were any missing values (there weren't :grinning:), and checked whether there was correlation between PRICE and the other variables (there was :grinning:). Following on from this I produced a heatmap to show the relative correlations of each of the variables. The intention was to avoid using variables that had little/no correlation with PRICE, as doing so would hinder the effectiveness of the model.

Here's the heatmap I produced:
![Boston ML - Heatmap](https://user-images.githubusercontent.com/84407701/160462696-5ca6bb1e-f110-4586-904d-daddfc468432.png)

##### Checking For Outliers
To check for outliers, I first produced a set of scatterplots for each variable against the main PRICE variable. This was done using a loop, with a subsplots used to designate the location of the plot.

These are the scatterplots I produced:
![Boston ML - Scatterplots](https://user-images.githubusercontent.com/84407701/160463897-04c98d69-9312-46a1-956c-d95b348a76e7.png)

From this I found that there were some potential outliers in the data, particularly in the CRIM and B variables, therefore I thought it would be worth taking a closer look to see whether this could be alleviated.

I decided to use Tukey's method to analyse the extent of outliers in each of the variables. To do this I amended an existing function created by [Alicia Horsch](https://aliciahorsch.medium.com/) which takes in the dataset and variable, and outputs the indexes of the probable outliers of the variable.

Following on from this, I created a loop which iterates through the variables and calculates the percentage of outliers in each case. The results were as follows:

- Percentage of CRIM observations that may be outliers: 13.04 %
- Percentage of ZN observations that may be outliers:  13.44 %
- Percentage of INDUS observations that may be outliers: 0.0 %
- Percentage of CHAS observations that may be outliers: 100.0 %
- Percentage of NOX observations that may be outliers: 0.0 %
- Percentage of RM observations that may be outliers: 5.93 %
- Percentage of AGE observations that may be outliers: 0.0 %
- Percentage of DIS observations that may be outliers: 0.99 %
- Percentage of RAD observations that may be outliers: 0.0 %
- Percentage of TAX observations that may be outliers: 0.0 %
- Percentage of PTRATIO observations that may be outliers: 2.96 %
- Percentage of B observations that may be outliers: 15.22 %
- Percentage of LSTAT observations that may be outliers: 1.38 %
 
The CHAS result should be ignored, as this comes from the fact that it is a dummy variable. More interesting, my intuition that there were quite a few outliers on the CRIM and B variables was correct. Curiously, there were quite a few outliers identified on the ZN variable. Using this information, I later experimented with removing these variables completley to see whether the fit of the model improved. 

Following on from this univariate approach, I employed the multivariate approach also suggested by [Alicia Horsch](https://aliciahorsch.medium.com/). This found outliers that were notable over multiple variables. Again I experimented with removing these outliers in the ML step. 

##### Machine Learning
All machine learning was done using the Sci-kit learn train_test_split function. This function seperates the dataset into a training and testing dataset and then fits a model to predict Y (PRICE) as well as possible. 

###### ML Model 1 - Raw Dataset
Best Results:

Training Set
- MAE: 3.258644946758388
- MSE: 20.800029437186076
- RMSE: 4.560704927660424
- R SQUARED 0.7451618941471037

Testing Set
- MAE: 3.351482441060827
- MSE: 24.604742571061884
- RMSE: 4.960316781321722
- R SQUARED 0.7236275652753374

An R-Squared of 0.72 in the test set is a good benchmark for the ML. In the following models I will attempt to better this by removing variables completely and experimenting with removing outliers as previously defined.

###### ML Model 2 - With B Variable
Best Results:

Training Set
- MAE: 3.437768399856977
- MSE: 22.190353331409465
- RMSE: 4.710663788831619
- R SQUARED 0.743975630070887

Testing Set
- MAE: 3.3264789277109084
- MSE: 23.676216522863797
- RMSE: 4.865821258828133
- R SQUARED 0.7009511790469423

This is a slight worsening in performance. Back to the drawing board! :laughing:

###### ML Model 3 - With CRIM Variable
Best Results:

Training Set
-MAE: 3.2964948463466115
-MSE: 21.420112027065425
-RMSE: 4.628186688873454
-R SQUARED 0.7375647571740809

Testing Set
-MAE: 3.3512530264642906
-MSE: 24.553899596340393
-RMSE: 4.955189158482287
-R SQUARED 0.7241986582941666

This is a very very slight worsening in performance over the standard model. Once again, back to the drawing board!!

###### ML Model 4 - With ZN Variable
Best Results:

Training Set
-MAE: 3.3685977677178496
-MSE: 21.624947517973744
-RMSE: 4.650263166528723
-R SQUARED 0.735055150701063

Testing Set
-MAE: 3.427291891997033
-MSE: 24.395400277112937
-RMSE: 4.939169998806777
-R SQUARED 0.7259789997316184

Once again, the performance of the ML model has worsened. But the knowledge gained from these experiments will help inform the following attempts at improving the model.

###### ML Model 5 - Without ZN/CRIM/B outliers
To check whether there was overlap with these outliers, I compared the number of outliers spotted in each instance. There were 58 B outliers, 30 CIRM outliesr and 45 ZN reported outliers spotted, therefore it made sense to experiment removing each for the ML model.

Without B
Training Set
-MAE: 3.2665323577082916
-MSE: 23.199661679619084
-RMSE: 4.816602711415909
-R SQUARED 0.7478541829112333

Testing Set
-MAE: 3.3960065796228283
-MSE: 20.328151164197628
-RMSE: 4.508675100758274
-R SQUARED 0.7034466653432716

Without CRIM
Training Set
-MAE: 3.277770626946739
-MSE: 21.889716074227845
-RMSE: 4.678644683477026
-R SQUARED 0.735906144074987

Testing Set
-MAE: 3.201831828546929
-MSE: 19.421710885882202
-RMSE: 4.407007021310744
-R SQUARED 0.7328859035925419

Without ZN
Training Set
-MAE: 3.2620054529188502
-MSE: 21.896449829158062
-RMSE: 4.679364254806209
-R SQUARED 0.7582785908749615

Testing Set
-MAE: 3.667510417595896
-MSE: 25.465273542310996
-RMSE: 5.046312866074694
-R SQUARED 0.5366993421033391

I found that removing CRIM helped the most, however, performance was still only marginally different to the raw dataset. I decided it was time to bring out the big guns :gun:

###### ML Model 6 - Without multivariate outliers
Best Results:

Training Set
- MAE: 2.679929018134156
- MSE: 12.92459729199948
- RMSE: 3.5950795946681735
- R SQUARED 0.8254228109293913

Testing Set
- MAE: 2.867149427663045
- MSE: 14.874127823928726
- RMSE: 3.8566990839225097
- R SQUARED 0.8190642641773782

These are the best results seen yet, with the R-Squared value increasing significantly from the raw ML at the start.

These are the values of the coefficients calculated in this model:
- Contant = 14.680121078331476
- CRIM       -0.081968
- ZN          0.024651
- INDUS       0.034646
- CHAS        2.298155
- NOX       -10.136647
- RM          6.061194
- AGE        -0.034978
- DIS        -1.095023
- RAD         0.185647
- TAX        -0.013007
- PTRATIO    -0.917359
- B           0.012717
- LSTAT      -0.248119
