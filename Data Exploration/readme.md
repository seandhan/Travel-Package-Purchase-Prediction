<font style = "font-family: Arial; font-weight:bold;font-size:2em;color:blue;">Context</font>

"**Visit with us**" travel company wants to retain its customers for a longer time period by launching a long-term travel package.  

The company had launched a holiday package last year and **18%** of the customers purchased that package however, *the marketing cost was quite high* because *customers were contacted at random without looking at the available information*.  

Now again the company is planning to launch a new product i.e., a long-term travel package, but this time company wants to *utilize previously available data* to reduce the marketing cost.  

You as a data scientist at "Visit with us" travel company have to *analyze the trend of existing customers' data* and information to *provide recommendations to the marketing team* and also *build a model to predict which customer is potentially going to purchase the long-term travel package*.


<font style = "font-family: Arial; font-weight:bold;font-size:2em;color:blue;">Objective</font>


1. To predict which customer is more likely to purchase the long-term travel package.

<font style = "font-family: Arial; font-weight:bold;font-size:2em;color:blue;">Data Dictionary</font>

**Customer details:**
1. **CustomerID**: Unique customer ID
2. **ProdTaken**: Product taken flag
3. **Age**: Age of customer
4. **PreferredLoginDevice**: Preferred login device of the customer in last month
5. **CityTier**: City tier
6. **Occupation**: Occupation of customer
7. **Gender**: Gender of customer
8. **NumberOfPersonVisited**: Total number of person came with customer
9. **PreferredPropertyStar**: Preferred hotel property rating by customer
10. **MaritalStatus**: Marital status of customer
11. **NumberOfTrips**: Average number of the trip in a year by customer
12. **Passport**: Customer passport flag
13. **OwnCar**: Customers owns a car flag
14. **NumberOfChildrenVisited**: Total number of children visit with customer
15. **Designation**: Designation of the customer in the current organization
16. **MonthlyIncome**: Gross monthly income of the customer

**Customer interaction data:**
1. **PitchSatisfactionScore**: Sales pitch satisfactory score
2. **ProductPitched**: Product pitched by a salesperson
3. **NumberOfFollowups**: Total number of follow up has been done by sales person after sales pitch
4. **DurationOfPitch**: Duration of the pitch by a salesman to customer


---

# Import all the necessary libraries


```python
# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as spy
from scipy import stats
# from scipy.stats import zscore, norm, randint
%matplotlib inline
import copy
```


```python
# Impute and Encode
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder
```


```python
# Modelling - Preparation, Metrics, Classifiers

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn import metrics, tree

from sklearn.tree import DecisionTreeClassifier

# Ensemble Methods Classifers
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier

#To install xgboost library use - !pip install xgboost
from xgboost import XGBClassifier

# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.metrics import precision_recall_curve

# import statsmodels.api as sm
# from statsmodels.tools.tools import add_constant
# from statsmodels.stats.outliers_influence import variance_inflation_factor
```


```python
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
```


```python
# Pandas display settings - rows & columns

# Display all rows
# pd.options.display.max_rows = 10000

# Display all columns
pd.set_option("display.max_columns", None)
```

# Data ingestion 

Initial there were issues reading the **xlsx** file as **XLRD** was modified in the most recent update, it could not read XLSX files.

I had to update Pandas library to the most recent version 1.2.1 (Jan 20, 2021)


```python
# Check to see Pandas version is 1.2.1
print("The version of Pandas library used in this notebook is: ", pd.__version__)

if pd.__version__ != "1.2.1":
    print("Pandas library need to be updated to version 1.2.1")
    # !pip install --upgrade pandas
```

    The version of Pandas library used in this notebook is:  1.1.3
    Pandas library need to be updated to version 1.2.1
    


```python
# Load dataset
data = pd.read_excel('Tourism.xlsx', sheet_name='Tourism')
```

# **Data Inspection**

**Preview dataset**


```python
# Preview the dataset
# View the first 5, last 5 and random 10 rows
print('First five rows', '--'*55)
display(data.head())

print('Last five rows', '--'*55)
display(data.tail())

print('Random ten rows', '--'*55)
np.random.seed(1)
display(data.sample(n=10))
```

    First five rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>ProdTaken</th>
      <th>Age</th>
      <th>PreferredLoginDevice</th>
      <th>CityTier</th>
      <th>DurationOfPitch</th>
      <th>Occupation</th>
      <th>Gender</th>
      <th>NumberOfPersonVisited</th>
      <th>NumberOfFollowups</th>
      <th>ProductPitched</th>
      <th>PreferredPropertyStar</th>
      <th>MaritalStatus</th>
      <th>NumberOfTrips</th>
      <th>Passport</th>
      <th>PitchSatisfactionScore</th>
      <th>OwnCar</th>
      <th>NumberOfChildrenVisited</th>
      <th>Designation</th>
      <th>MonthlyIncome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200000</td>
      <td>1</td>
      <td>41.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>6.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>3</td>
      <td>3.0</td>
      <td>Super Deluxe</td>
      <td>3.0</td>
      <td>Single</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>Manager</td>
      <td>20993.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200001</td>
      <td>0</td>
      <td>49.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>14.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>Super Deluxe</td>
      <td>4.0</td>
      <td>Divorced</td>
      <td>2.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2.0</td>
      <td>Manager</td>
      <td>20130.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200002</td>
      <td>1</td>
      <td>37.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>8.0</td>
      <td>Free Lancer</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Single</td>
      <td>7.0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>Executive</td>
      <td>17090.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200003</td>
      <td>0</td>
      <td>33.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>9.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>2</td>
      <td>3.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>2.0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>17909.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200004</td>
      <td>0</td>
      <td>NaN</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>8.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>2</td>
      <td>3.0</td>
      <td>Multi</td>
      <td>4.0</td>
      <td>Divorced</td>
      <td>1.0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>Executive</td>
      <td>18468.0</td>
    </tr>
  </tbody>
</table>
</div>


    Last five rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>ProdTaken</th>
      <th>Age</th>
      <th>PreferredLoginDevice</th>
      <th>CityTier</th>
      <th>DurationOfPitch</th>
      <th>Occupation</th>
      <th>Gender</th>
      <th>NumberOfPersonVisited</th>
      <th>NumberOfFollowups</th>
      <th>ProductPitched</th>
      <th>PreferredPropertyStar</th>
      <th>MaritalStatus</th>
      <th>NumberOfTrips</th>
      <th>Passport</th>
      <th>PitchSatisfactionScore</th>
      <th>OwnCar</th>
      <th>NumberOfChildrenVisited</th>
      <th>Designation</th>
      <th>MonthlyIncome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4883</th>
      <td>204883</td>
      <td>1</td>
      <td>49.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>9.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>3</td>
      <td>5.0</td>
      <td>Super Deluxe</td>
      <td>4.0</td>
      <td>Unmarried</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>Manager</td>
      <td>26576.0</td>
    </tr>
    <tr>
      <th>4884</th>
      <td>204884</td>
      <td>1</td>
      <td>28.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>31.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>5.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Single</td>
      <td>3.0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2.0</td>
      <td>Executive</td>
      <td>21212.0</td>
    </tr>
    <tr>
      <th>4885</th>
      <td>204885</td>
      <td>1</td>
      <td>52.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>17.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>4</td>
      <td>4.0</td>
      <td>Standard</td>
      <td>4.0</td>
      <td>Married</td>
      <td>7.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>Senior Manager</td>
      <td>31820.0</td>
    </tr>
    <tr>
      <th>4886</th>
      <td>204886</td>
      <td>1</td>
      <td>19.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>16.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Single</td>
      <td>3.0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>2.0</td>
      <td>Executive</td>
      <td>20289.0</td>
    </tr>
    <tr>
      <th>4887</th>
      <td>204887</td>
      <td>1</td>
      <td>36.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>14.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>4.0</td>
      <td>Unmarried</td>
      <td>3.0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2.0</td>
      <td>Executive</td>
      <td>24041.0</td>
    </tr>
  </tbody>
</table>
</div>


    Random ten rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>ProdTaken</th>
      <th>Age</th>
      <th>PreferredLoginDevice</th>
      <th>CityTier</th>
      <th>DurationOfPitch</th>
      <th>Occupation</th>
      <th>Gender</th>
      <th>NumberOfPersonVisited</th>
      <th>NumberOfFollowups</th>
      <th>ProductPitched</th>
      <th>PreferredPropertyStar</th>
      <th>MaritalStatus</th>
      <th>NumberOfTrips</th>
      <th>Passport</th>
      <th>PitchSatisfactionScore</th>
      <th>OwnCar</th>
      <th>NumberOfChildrenVisited</th>
      <th>Designation</th>
      <th>MonthlyIncome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3015</th>
      <td>203015</td>
      <td>0</td>
      <td>27.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>7.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>4</td>
      <td>6.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Married</td>
      <td>5.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>3.0</td>
      <td>Executive</td>
      <td>23042.0</td>
    </tr>
    <tr>
      <th>1242</th>
      <td>201242</td>
      <td>0</td>
      <td>40.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>13.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>2</td>
      <td>3.0</td>
      <td>King</td>
      <td>4.0</td>
      <td>Single</td>
      <td>2.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>NaN</td>
      <td>VP</td>
      <td>34833.0</td>
    </tr>
    <tr>
      <th>3073</th>
      <td>203073</td>
      <td>0</td>
      <td>29.0</td>
      <td>Self Enquiry</td>
      <td>2</td>
      <td>15.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>4</td>
      <td>5.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2.0</td>
      <td>Executive</td>
      <td>23614.0</td>
    </tr>
    <tr>
      <th>804</th>
      <td>200804</td>
      <td>0</td>
      <td>48.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>6.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>2</td>
      <td>1.0</td>
      <td>Deluxe</td>
      <td>3.0</td>
      <td>Single</td>
      <td>3.0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>AVP</td>
      <td>31885.0</td>
    </tr>
    <tr>
      <th>3339</th>
      <td>203339</td>
      <td>0</td>
      <td>32.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>18.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>4</td>
      <td>4.0</td>
      <td>Super Deluxe</td>
      <td>5.0</td>
      <td>Divorced</td>
      <td>3.0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3.0</td>
      <td>Manager</td>
      <td>25511.0</td>
    </tr>
    <tr>
      <th>3080</th>
      <td>203080</td>
      <td>1</td>
      <td>36.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>32.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>4</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>4.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>20700.0</td>
    </tr>
    <tr>
      <th>2851</th>
      <td>202851</td>
      <td>0</td>
      <td>46.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>17.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>5.0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>21332.0</td>
    </tr>
    <tr>
      <th>2883</th>
      <td>202883</td>
      <td>1</td>
      <td>32.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>27.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>4.0</td>
      <td>Standard</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>5.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>Senior Manager</td>
      <td>28502.0</td>
    </tr>
    <tr>
      <th>1676</th>
      <td>201676</td>
      <td>0</td>
      <td>22.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>11.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>2</td>
      <td>1.0</td>
      <td>Multi</td>
      <td>4.0</td>
      <td>Married</td>
      <td>2.0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>Executive</td>
      <td>17328.0</td>
    </tr>
    <tr>
      <th>1140</th>
      <td>201140</td>
      <td>0</td>
      <td>44.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>13.0</td>
      <td>Small Business</td>
      <td>Female</td>
      <td>2</td>
      <td>3.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>1.0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1.0</td>
      <td>VP</td>
      <td>34049.0</td>
    </tr>
  </tbody>
</table>
</div>


- `CustomerID` is row identifier, which does not add any value. This variable will be removed later.
- There are missing values in the dataset as indicated by **Nan** in the `Age` variable.

## Variable List


```python
# Display list of variables in dataset
variable_list = data.columns.tolist()
print(variable_list)
```

    ['CustomerID', 'ProdTaken', 'Age', 'PreferredLoginDevice', 'CityTier', 'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisited', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisited', 'Designation', 'MonthlyIncome']
    

## Dataset shape


```python
shape = data.shape
n_rows = shape[0]
n_cols = shape[1]
print(f"The Dataframe consists of '{n_rows}' rows and '{n_cols}' columns")
```

    The Dataframe consists of '4888' rows and '20' columns
    

**Data types**


```python
# Check the data types
data.dtypes
```




    CustomerID                   int64
    ProdTaken                    int64
    Age                        float64
    PreferredLoginDevice        object
    CityTier                     int64
    DurationOfPitch            float64
    Occupation                  object
    Gender                      object
    NumberOfPersonVisited        int64
    NumberOfFollowups          float64
    ProductPitched              object
    PreferredPropertyStar      float64
    MaritalStatus               object
    NumberOfTrips              float64
    Passport                     int64
    PitchSatisfactionScore       int64
    OwnCar                       int64
    NumberOfChildrenVisited    float64
    Designation                 object
    MonthlyIncome              float64
    dtype: object



**Data info**


```python
# Get info of the dataframe columns
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4888 entries, 0 to 4887
    Data columns (total 20 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   CustomerID               4888 non-null   int64  
     1   ProdTaken                4888 non-null   int64  
     2   Age                      4662 non-null   float64
     3   PreferredLoginDevice     4863 non-null   object 
     4   CityTier                 4888 non-null   int64  
     5   DurationOfPitch          4637 non-null   float64
     6   Occupation               4888 non-null   object 
     7   Gender                   4888 non-null   object 
     8   NumberOfPersonVisited    4888 non-null   int64  
     9   NumberOfFollowups        4843 non-null   float64
     10  ProductPitched           4888 non-null   object 
     11  PreferredPropertyStar    4862 non-null   float64
     12  MaritalStatus            4888 non-null   object 
     13  NumberOfTrips            4748 non-null   float64
     14  Passport                 4888 non-null   int64  
     15  PitchSatisfactionScore   4888 non-null   int64  
     16  OwnCar                   4888 non-null   int64  
     17  NumberOfChildrenVisited  4822 non-null   float64
     18  Designation              4888 non-null   object 
     19  MonthlyIncome            4655 non-null   float64
    dtypes: float64(7), int64(7), object(6)
    memory usage: 763.9+ KB
    

- Six (6) variables have been identified as `Panda object` type. These shall be converted to the `category` type.

**Convert Pandas Objects to Category type**


```python
# Convert variables with "object" type to "category" type
for i in data.columns:
    if data[i].dtypes == "object":
        data[i] = data[i].astype("category") 

# Confirm if there no variables with "object" type
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4888 entries, 0 to 4887
    Data columns (total 20 columns):
     #   Column                   Non-Null Count  Dtype   
    ---  ------                   --------------  -----   
     0   CustomerID               4888 non-null   int64   
     1   ProdTaken                4888 non-null   int64   
     2   Age                      4662 non-null   float64 
     3   PreferredLoginDevice     4863 non-null   category
     4   CityTier                 4888 non-null   int64   
     5   DurationOfPitch          4637 non-null   float64 
     6   Occupation               4888 non-null   category
     7   Gender                   4888 non-null   category
     8   NumberOfPersonVisited    4888 non-null   int64   
     9   NumberOfFollowups        4843 non-null   float64 
     10  ProductPitched           4888 non-null   category
     11  PreferredPropertyStar    4862 non-null   float64 
     12  MaritalStatus            4888 non-null   category
     13  NumberOfTrips            4748 non-null   float64 
     14  Passport                 4888 non-null   int64   
     15  PitchSatisfactionScore   4888 non-null   int64   
     16  OwnCar                   4888 non-null   int64   
     17  NumberOfChildrenVisited  4822 non-null   float64 
     18  Designation              4888 non-null   category
     19  MonthlyIncome            4655 non-null   float64 
    dtypes: category(6), float64(7), int64(7)
    memory usage: 564.4 KB
    

- `The memory usage has decreased from 764 KB to 565 KB`

**Missing value summary function**


```python
def missing_val_chk(data):
    """
    This function to checks for missing values 
    and generates a summary.
    """
    if data.isnull().sum().any() == True:
        # Number of missing in each column
        missing_vals = pd.DataFrame(data.isnull().sum().sort_values(
            ascending=False)).rename(columns={0: '# missing'})

        # Create a percentage missing
        missing_vals['percent'] = ((missing_vals['# missing'] / len(data)) *
                                   100).round(decimals=3)

        # Remove rows with 0
        missing_vals = missing_vals[missing_vals['# missing'] != 0].dropna()

        # display missing value dataframe
        print("The missing values summary")
        display(missing_vals)
    else:
        print("There are NO missing values in the dataset")
```

## Missing Values Check


```python
#Applying the missing value summary function
missing_val_chk(data)
```

    The missing values summary
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># missing</th>
      <th>percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DurationOfPitch</th>
      <td>251</td>
      <td>5.135</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>233</td>
      <td>4.767</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>226</td>
      <td>4.624</td>
    </tr>
    <tr>
      <th>NumberOfTrips</th>
      <td>140</td>
      <td>2.864</td>
    </tr>
    <tr>
      <th>NumberOfChildrenVisited</th>
      <td>66</td>
      <td>1.350</td>
    </tr>
    <tr>
      <th>NumberOfFollowups</th>
      <td>45</td>
      <td>0.921</td>
    </tr>
    <tr>
      <th>PreferredPropertyStar</th>
      <td>26</td>
      <td>0.532</td>
    </tr>
    <tr>
      <th>PreferredLoginDevice</th>
      <td>25</td>
      <td>0.511</td>
    </tr>
  </tbody>
</table>
</div>


***

## 5 Point Summary

**Numerical type Summary**


```python
# Five point summary of all numerical type variables in the dataset
data.describe().T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CustomerID</th>
      <td>4888.0</td>
      <td>202443.500000</td>
      <td>1411.188388</td>
      <td>200000.0</td>
      <td>201221.75</td>
      <td>202443.5</td>
      <td>203665.25</td>
      <td>204887.0</td>
    </tr>
    <tr>
      <th>ProdTaken</th>
      <td>4888.0</td>
      <td>0.188216</td>
      <td>0.390925</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>4662.0</td>
      <td>37.622265</td>
      <td>9.316387</td>
      <td>18.0</td>
      <td>31.00</td>
      <td>36.0</td>
      <td>44.00</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>CityTier</th>
      <td>4888.0</td>
      <td>1.654255</td>
      <td>0.916583</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>3.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>DurationOfPitch</th>
      <td>4637.0</td>
      <td>15.490835</td>
      <td>8.519643</td>
      <td>5.0</td>
      <td>9.00</td>
      <td>13.0</td>
      <td>20.00</td>
      <td>127.0</td>
    </tr>
    <tr>
      <th>NumberOfPersonVisited</th>
      <td>4888.0</td>
      <td>2.905074</td>
      <td>0.724891</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>NumberOfFollowups</th>
      <td>4843.0</td>
      <td>3.708445</td>
      <td>1.002509</td>
      <td>1.0</td>
      <td>3.00</td>
      <td>4.0</td>
      <td>4.00</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>PreferredPropertyStar</th>
      <td>4862.0</td>
      <td>3.581037</td>
      <td>0.798009</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>NumberOfTrips</th>
      <td>4748.0</td>
      <td>3.236521</td>
      <td>1.849019</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>Passport</th>
      <td>4888.0</td>
      <td>0.290917</td>
      <td>0.454232</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>PitchSatisfactionScore</th>
      <td>4888.0</td>
      <td>3.078151</td>
      <td>1.365792</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>OwnCar</th>
      <td>4888.0</td>
      <td>0.620295</td>
      <td>0.485363</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>NumberOfChildrenVisited</th>
      <td>4822.0</td>
      <td>1.187267</td>
      <td>0.857861</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>4655.0</td>
      <td>23619.853491</td>
      <td>5380.698361</td>
      <td>1000.0</td>
      <td>20346.00</td>
      <td>22347.0</td>
      <td>25571.00</td>
      <td>98678.0</td>
    </tr>
  </tbody>
</table>
</div>



- `ProdTaken` is a binary variable with 18.8% of the rows having a value of 1
- `Age` is fairly symmetrical with *mean* and *median* being very close
- `CityTier` is a categorical ordinal variable with three states
- `DurationOfPitch` is numerical variable being highly right skewed as there is significant change between Q3 and Q4
- `NumberOfPersonVisited` is a categorical ordinal variable with five states
- `PreferredPropertyStar` is a categorical ordinal variable with three states
- `NumberOfTrips` is numerical variable being highly right skewed as there is significant change between Q3 and Q4
- `Passport` is a binary variable with 29.1% of the rows having a value of 1
- `PitchSatisfactionScore` is a categorical ordinal variable with five states
- `OwnCar` is a binary variable with 62% of the rows having a value of 1
- `NumberOfChildrenVisited` is a categorical ordinal variable with three states
- `MonthlyIncome` is numerical variable being highly right skewed as there is significant change between Q3 and Q4

**Categorical type Summary**


```python
data.describe(include=['category']).T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PreferredLoginDevice</th>
      <td>4863</td>
      <td>2</td>
      <td>Self Enquiry</td>
      <td>3444</td>
    </tr>
    <tr>
      <th>Occupation</th>
      <td>4888</td>
      <td>4</td>
      <td>Salaried</td>
      <td>2368</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>4888</td>
      <td>3</td>
      <td>Male</td>
      <td>2916</td>
    </tr>
    <tr>
      <th>ProductPitched</th>
      <td>4888</td>
      <td>5</td>
      <td>Multi</td>
      <td>1842</td>
    </tr>
    <tr>
      <th>MaritalStatus</th>
      <td>4888</td>
      <td>4</td>
      <td>Married</td>
      <td>2340</td>
    </tr>
    <tr>
      <th>Designation</th>
      <td>4888</td>
      <td>5</td>
      <td>Executive</td>
      <td>1842</td>
    </tr>
  </tbody>
</table>
</div>



- `Gender` has three states which seems a bit odd. Further investigation will be done.

---

**Number of unique states for all variables**


```python
# Check the unique values
data.nunique().to_frame()
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CustomerID</th>
      <td>4888</td>
    </tr>
    <tr>
      <th>ProdTaken</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>44</td>
    </tr>
    <tr>
      <th>PreferredLoginDevice</th>
      <td>2</td>
    </tr>
    <tr>
      <th>CityTier</th>
      <td>3</td>
    </tr>
    <tr>
      <th>DurationOfPitch</th>
      <td>34</td>
    </tr>
    <tr>
      <th>Occupation</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>3</td>
    </tr>
    <tr>
      <th>NumberOfPersonVisited</th>
      <td>5</td>
    </tr>
    <tr>
      <th>NumberOfFollowups</th>
      <td>6</td>
    </tr>
    <tr>
      <th>ProductPitched</th>
      <td>5</td>
    </tr>
    <tr>
      <th>PreferredPropertyStar</th>
      <td>3</td>
    </tr>
    <tr>
      <th>MaritalStatus</th>
      <td>4</td>
    </tr>
    <tr>
      <th>NumberOfTrips</th>
      <td>12</td>
    </tr>
    <tr>
      <th>Passport</th>
      <td>2</td>
    </tr>
    <tr>
      <th>PitchSatisfactionScore</th>
      <td>5</td>
    </tr>
    <tr>
      <th>OwnCar</th>
      <td>2</td>
    </tr>
    <tr>
      <th>NumberOfChildrenVisited</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Designation</th>
      <td>5</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>2475</td>
    </tr>
  </tbody>
</table>
</div>



* `Age`, `DurationOfPitch`, `NumberOfTrips` & `MonthlyIncome` are numerical variables

---

**Categorical Variable Identification**

Although the following variables are numerical in nature, they represent **categorical** variables:
* `CustomerID`
* `ProdTaken`
* `PreferredLoginDevice`
* `CityTier`
* `Occupation`
* `Gender`
* `NumberOfPersonVisited`
* `NumberOfFollowups` 
* `ProductPitched`
* `PreferredPropertyStar`
* `MaritalStatus`
* `Passport`
* `PitchSatisfactionScore`
* `OwnCar`
* `NumberOfChildrenVisited`
* `Designation`

---

**Create a list of numerical variables**


```python
numerical_vars = ['Age', 'DurationOfPitch', 'NumberOfTrips', 'MonthlyIncome']
```

**Create a list of categorical variables**


```python
categorical_vars = [
    'CustomerID', 'ProdTaken', 'PreferredLoginDevice', 'CityTier',
    'Occupation', 'Gender', 'NumberOfPersonVisited', 'NumberOfFollowups',
    'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisited',
    'Designation'
]
```

---

## Numerical data


```python
data[numerical_vars].describe().T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>4662.0</td>
      <td>37.622265</td>
      <td>9.316387</td>
      <td>18.0</td>
      <td>31.0</td>
      <td>36.0</td>
      <td>44.0</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>DurationOfPitch</th>
      <td>4637.0</td>
      <td>15.490835</td>
      <td>8.519643</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>20.0</td>
      <td>127.0</td>
    </tr>
    <tr>
      <th>NumberOfTrips</th>
      <td>4748.0</td>
      <td>3.236521</td>
      <td>1.849019</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>4655.0</td>
      <td>23619.853491</td>
      <td>5380.698361</td>
      <td>1000.0</td>
      <td>20346.0</td>
      <td>22347.0</td>
      <td>25571.0</td>
      <td>98678.0</td>
    </tr>
  </tbody>
</table>
</div>



### Skew Summary


```python
# Display the skew summary for the numerical variables
for var in data[numerical_vars].columns:
    var_skew = data[var].skew()
    if var_skew > 1:
        print(f"The '{var}' distribution is highly right skewed.\n")
    elif var_skew < -1:
        print(f"The '{var}' distribution is highly left skewed.\n")
    elif (var_skew > 0.5) & (var_skew < 1):
        print(f"The '{var}' distribution is moderately right skewed.\n")
    elif (var_skew < -0.5) & (var_skew > -1):
        print(f"The '{var}' distribution is moderately left skewed.\n")
    else:
        print(f"The '{var}' distribution is fairly symmetrical.\n")
```

    The 'Age' distribution is fairly symmetrical.
    
    The 'DurationOfPitch' distribution is highly right skewed.
    
    The 'NumberOfTrips' distribution is highly right skewed.
    
    The 'MonthlyIncome' distribution is highly right skewed.
    
    

**Outlier check function**


```python
# Outlier check
def outlier_count(data):
    """
    This function checks the lower and upper 
    outliers for all numerical variables.
    
    Outliers are found where data points exists either:
    - Greater than `1.5*IQR` above the 75th percentile
    - Less than `1.5*IQR` below the 25th percentile
    """
    numeric = data.select_dtypes(include=np.number).columns.to_list()
    for i in numeric:
        # Get name of series
        name = data[i].name
        # Calculate the IQR for all values and omit NaNs
        IQR = spy.stats.iqr(data[i], nan_policy="omit")
        # Calculate the boxplot upper fence
        upper_fence = data[i].quantile(0.75) + 1.5 * IQR
        # Calculate the boxplot lower fence
        lower_fence = data[i].quantile(0.25) - 1.5 * IQR
        # Calculate the count of outliers above upper fence
        upper_outliers = data[i][data[i] > upper_fence].count()
        # Calculate the count of outliers below lower fence
        lower_outliers = data[i][data[i] < lower_fence].count()
        # Check if there are no outliers
        if (upper_outliers == 0) & (lower_outliers == 0):
            continue
        print(
            f"The '{name}' distribution has '{lower_outliers}' lower outliers and '{upper_outliers}' upper outliers.\n"
        )
```

### Outlier check


```python
#Applying the Outlier check function for the sub-dataframe of numerical variables
outlier_count(data[numerical_vars])
```

    The 'DurationOfPitch' distribution has '0' lower outliers and '2' upper outliers.
    
    The 'NumberOfTrips' distribution has '0' lower outliers and '109' upper outliers.
    
    The 'MonthlyIncome' distribution has '2' lower outliers and '343' upper outliers.
    
    

### Numerical Variable Summary

| Variable| Skew | Outliers | 
| :-: | :-: | :-: |
| **Age** | Fairly symmetrical | No Outliers | 
| **DurationOfPitch** | Highly right skewed | 2 Upper Outliers | 
| **NumberOfTrips** | Highly right skewed | 109 Upper Outliers |
| **MonthlyIncome** | Highly right skewed | 2 Lower & 343 Upper Outliers |

---

## Categorical data

### Unique states

**Detailed investigation of unique values**


```python
# Display the unique values for all categorical variables
for i in categorical_vars:
    print('Unique values in',i, 'are :')
    print(data[i].value_counts())
    print('--'*55)
```

    Unique values in CustomerID are :
    200702    1
    201479    1
    203514    1
    201467    1
    203518    1
             ..
    204257    1
    200163    1
    202212    1
    204261    1
    204800    1
    Name: CustomerID, Length: 4888, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in ProdTaken are :
    0    3968
    1     920
    Name: ProdTaken, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in PreferredLoginDevice are :
    Self Enquiry       3444
    Company Invited    1419
    Name: PreferredLoginDevice, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in CityTier are :
    1    3190
    3    1500
    2     198
    Name: CityTier, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Occupation are :
    Salaried          2368
    Small Business    2084
    Large Business     434
    Free Lancer          2
    Name: Occupation, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Gender are :
    Male       2916
    Female     1817
    Fe Male     155
    Name: Gender, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in NumberOfPersonVisited are :
    3    2402
    2    1418
    4    1026
    1      39
    5       3
    Name: NumberOfPersonVisited, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in NumberOfFollowups are :
    4.0    2068
    3.0    1466
    5.0     768
    2.0     229
    1.0     176
    6.0     136
    Name: NumberOfFollowups, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in ProductPitched are :
    Multi           1842
    Super Deluxe    1732
    Standard         742
    Deluxe           342
    King             230
    Name: ProductPitched, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in PreferredPropertyStar are :
    3.0    2993
    5.0     956
    4.0     913
    Name: PreferredPropertyStar, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in MaritalStatus are :
    Married      2340
    Divorced      950
    Single        916
    Unmarried     682
    Name: MaritalStatus, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Passport are :
    0    3466
    1    1422
    Name: Passport, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in PitchSatisfactionScore are :
    3    1478
    5     970
    1     942
    4     912
    2     586
    Name: PitchSatisfactionScore, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in OwnCar are :
    1    3032
    0    1856
    Name: OwnCar, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in NumberOfChildrenVisited are :
    1.0    2080
    2.0    1335
    0.0    1082
    3.0     325
    Name: NumberOfChildrenVisited, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Designation are :
    Executive         1842
    Manager           1732
    Senior Manager     742
    AVP                342
    VP                 230
    Name: Designation, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    

- `Gender` -  There is another state **"Fe Male"**. This will be interpreted as a error in data input. All instances of **"Fe Male"** will be **replaced** by **"Female"**
- `MaritalStatus` - There are two states **"Single"** and **"Unmarried"** which are similar in certain contexts but will be left unchanged as such in the EDA.

---

**Replacing "Fe Male" with "Female"**


```python
# Replace "Fe Male" with "Female"
data['Gender'] = data['Gender'].replace({'Fe Male':'Female'})
```


```python
# Check states in "Gender"
data['Gender'].value_counts()
```




    Male      2916
    Female    1972
    Name: Gender, dtype: int64



---

### Categorical Variable Summary

There are categorical variables in the numeric format.

| Variable| Type | Range | 
| :-: | :-: | :-: |
| **CustomerID** |  Nominal | 200000-204887 |
| **ProdTaken**| Nominal | Binary |
| **PreferredLoginDevice**| Nominal | 2 states |
| **CityTier**| Ordinal | 3 states |
| **Occupation**| Nominal | 4 states |
| **Gender**| Nominal | 2 states |
| **NumberOfPersonVisited**| Ordinal | 5 states |
| **NumberOfFollowups**| Ordinal | 6 states |
| **ProductPitched**| Nominal | 5 states |
| **PreferredPropertyStar**| Ordinal | 3 states |
| **MaritalStatus**| Nominal | 4 states |
| **Passport**| Nominal | Binary |
| **PitchSatisfactionScore**| Ordinal | 5 states |
| **OwnCar**| Nominal | Binary |
| **NumberOfChildrenVisited**| Ordinal | 4 states |
| **Designation**| Nominal | 5 states |

---

## Target Variable

Target variable is **`ProdTaken`**


```python
# Checking the distribution of target variable

# Count the different "ProdTaken" states
count = data["ProdTaken"].value_counts().T
# Calculate the percentage different "ProdTaken" states
percentage = data['ProdTaken'].value_counts(normalize=True).T * 100
# Join count and percentage series
target_dist = pd.concat([count, percentage], axis=1)
# Set column names
target_dist.columns = ['count', 'percentage']
# Set Index name
target_dist.index.name = "ProdTaken"
# Display target distribution dataframe
target_dist
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>percentage</th>
    </tr>
    <tr>
      <th>ProdTaken</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3968</td>
      <td>81.178396</td>
    </tr>
    <tr>
      <th>1</th>
      <td>920</td>
      <td>18.821604</td>
    </tr>
  </tbody>
</table>
</div>



**Out of the 4888 customers, only 18.8% accepted the personal loan offer in the previous campaign**

<font color='red'> The Target variable is **Moderately Imbalanced**

---

**Dropping the `CustomerID` variable**

We shall drop the `CustomerID` variable as it does not add any value to the dataset.


```python
# Drop CustomerID column inplace
data.drop(columns = 'CustomerID', inplace=True)

# Remove CustomerID from "categorical_vars" list
categorical_vars.remove('CustomerID')
```

---

---
