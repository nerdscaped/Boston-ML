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
The first step was to load in the packages needed for the project. These included Pandas, NumPy, Matplotlib, Seaborn, Sci-kit learn and SciPy.

After this, I loaded in the dataset and performed some preliminary exploratory data analysis. For example, I checked whether there were any missing values (there weren't!), and checked whether there was correlation between PRICE and the other variables (there was!). Following on from this I produced a heatmap to show the relative correlations of each of the variables. The intention was to avoid using variables that had little/no correlation with PRICE, as doing so would hinder the effectiveness of the model.

Here's the heatmap I produced:

