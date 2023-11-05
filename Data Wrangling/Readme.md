# Missing Value Treatment

## Numerical data

**Missing Value Check for numerical variables**


```python
missing_val_chk(data[numerical_vars])
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
  </tbody>
</table>
</div>


Since the missing value % across the numerical variables are **~=< 5%**, let's impute the **median** of each variable in the missing values

Using the **SKLearn Simple Imputer** with **median** strategy


```python
# Declare numerical imputer function with median strategy
numerical_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

# Execute numerical imputer function on numerical variables
data[numerical_vars] = numerical_imputer.fit_transform(data[numerical_vars])

# Confirm if there are any missing values after impution
missing_val_chk(data[numerical_vars])
```

    There are NO missing values in the dataset
    

## Categorical data

**Missing Value Check for categorical variables**


```python
missing_val_chk(data[categorical_vars])
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


Since the missing value % across the categorical variables are **~=< 1%**, let's impute the **'most_frequent'** of each variable in the missing values

Using the **SKLearn Simple Imputer** with **median** strategy


```python
# Declare categorical imputer function with most_frequent strategy
categorical_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# Execute categorical imputer function on categorical variables
data[categorical_vars] = categorical_imputer.fit_transform(data[categorical_vars])

# Confirm if there are any missing values after impution
missing_val_chk(data[categorical_vars])
```

    There are NO missing values in the dataset
    

---

# Outlier Treatment

In the numerical data analysis, we have observed that`DurationOfPitch`, `NumberOfTrips` & `MonthlyIncome` variables have outliers.

According to literature, **Bagging** Ensemble methods are robust to outliers while **Boosting** Ensemble methods may be sensitive.  
Therefore outlier treatment will only be necessary for the **Boosting** Ensemble methods.

Before we decide on whether or not Outlier Treatment be done prior EDA, lets explore the variables.

<!-- Therefore in further analysis, I shall run the Bagging methods with the data as such. For the Boosting methods, I will clone the data, deal with the outliers the data and compare model performances with & without outliers. -->

---

## Outlier exploration


```python
#Applying the Outlier check function for the sub-dataframe of numerical variables
outlier_count(data[numerical_vars])
```

    The 'DurationOfPitch' distribution has '0' lower outliers and '112' upper outliers.
    
    The 'NumberOfTrips' distribution has '0' lower outliers and '109' upper outliers.
    
    The 'MonthlyIncome' distribution has '2' lower outliers and '373' upper outliers.
    
    


```python
def quantile_check(variable):
    """
    This function to explores the variable values at
    predefined quantile intervals.
    This is will aid in determining capping limits.
    """
    quantile_range = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0]
    for quantile in quantile_range:
        interval = variable.quantile(quantile)
        print(f"The '{quantile}' quantile value is '{interval}'.")
```

---

### DurationOfPitch


```python
data.DurationOfPitch.describe()
```




    count    4888.000000
    mean       15.362930
    std         8.316166
    min         5.000000
    25%         9.000000
    50%        13.000000
    75%        19.000000
    max       127.000000
    Name: DurationOfPitch, dtype: float64



Between Q3 and Q4, there is significant change compared to (Q1-Q2) & (Q2-Q3).  
Let's explore further to see where there is a rapid jump.


```python
quantile_check(data.DurationOfPitch)
```

    The '0.8' quantile value is '22.0'.
    The '0.9' quantile value is '29.0'.
    The '0.95' quantile value is '32.0'.
    The '0.98' quantile value is '35.0'.
    The '0.99' quantile value is '35.0'.
    The '0.995' quantile value is '36.0'.
    The '1.0' quantile value is '127.0'.
    

There is a rapid jump after the 99.5 percentile.
Let's explore these rows


```python
data[data.DurationOfPitch>data.DurationOfPitch.quantile(0.995)]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>1434</th>
      <td>0</td>
      <td>36.0</td>
      <td>Company Invited</td>
      <td>3</td>
      <td>126.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>2</td>
      <td>3.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>18482.0</td>
    </tr>
    <tr>
      <th>3878</th>
      <td>0</td>
      <td>53.0</td>
      <td>Company Invited</td>
      <td>3</td>
      <td>127.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Married</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>Executive</td>
      <td>22160.0</td>
    </tr>
  </tbody>
</table>
</div>



Observation:
* There are 2 Married Male customers, who are both 'Executive' of significantly different Age groups

Since there are only *2* rows and the difference in value between the 99.5 and 100 percentiles is very large, let's cap the values at the 99.5 percentile


```python
# capping outliers at 99.5 percentile

data.DurationOfPitch = np.where(
    data.DurationOfPitch > data.DurationOfPitch.quantile(0.995),
    data.DurationOfPitch.quantile(0.995), data.DurationOfPitch)
```

---

### NumberOfTrips


```python
data.NumberOfTrips.describe()
```




    count    4888.000000
    mean        3.229746
    std         1.822769
    min         1.000000
    25%         2.000000
    50%         3.000000
    75%         4.000000
    max        22.000000
    Name: NumberOfTrips, dtype: float64



Between Q3 and Q4, there is significant change compared to (Q1-Q2) & (Q2-Q3).  
Let's explore further to see where there is a rapid jump.


```python
quantile_check(data.NumberOfTrips)
```

    The '0.8' quantile value is '5.0'.
    The '0.9' quantile value is '6.0'.
    The '0.95' quantile value is '7.0'.
    The '0.98' quantile value is '8.0'.
    The '0.99' quantile value is '8.0'.
    The '0.995' quantile value is '8.0'.
    The '1.0' quantile value is '22.0'.
    

There is a rapid jump after the 99.5 percentile.
Let's explore these rows


```python
data[data.NumberOfTrips>data.NumberOfTrips.quantile(0.995)]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>385</th>
      <td>1</td>
      <td>30.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>10.0</td>
      <td>Large Business</td>
      <td>Male</td>
      <td>2</td>
      <td>3.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Single</td>
      <td>19.0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>17285.0</td>
    </tr>
    <tr>
      <th>816</th>
      <td>0</td>
      <td>39.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>15.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>3</td>
      <td>3.0</td>
      <td>Super Deluxe</td>
      <td>4.0</td>
      <td>Unmarried</td>
      <td>21.0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>Manager</td>
      <td>21782.0</td>
    </tr>
    <tr>
      <th>2829</th>
      <td>1</td>
      <td>31.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>11.0</td>
      <td>Large Business</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Single</td>
      <td>20.0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2.0</td>
      <td>Executive</td>
      <td>20963.0</td>
    </tr>
    <tr>
      <th>3260</th>
      <td>0</td>
      <td>40.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>16.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>4.0</td>
      <td>Super Deluxe</td>
      <td>4.0</td>
      <td>Unmarried</td>
      <td>22.0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1.0</td>
      <td>Manager</td>
      <td>25460.0</td>
    </tr>
  </tbody>
</table>
</div>



Observation:
* There are 4 Male customers, who are 'Executive' & 'Manager' of similar Age groups. They are 'Single' and 'Unmarried'

Since there are only *4* rows and the difference in value between the 99.5 and 100 percentiles is very large, let's cap the values at the 99.5 percentile


```python
# capping outliers at 99.5 percentile

data.NumberOfTrips = np.where(
    data.NumberOfTrips > data.NumberOfTrips.quantile(0.995),
    data.NumberOfTrips.quantile(0.995), data.NumberOfTrips)
```

---

### MonthlyIncome


```python
data.MonthlyIncome.describe()
```




    count     4888.000000
    mean     23559.179419
    std       5257.862921
    min       1000.000000
    25%      20485.000000
    50%      22347.000000
    75%      25424.750000
    max      98678.000000
    Name: MonthlyIncome, dtype: float64



Between (Q0 - Q1) & (Q3 - Q4), there is significant change compared to (Q1-Q2) & (Q2-Q3).  
Let's explore further to see where there is a rapid jump.

**Lower Outliers**


```python
def lower_quantile_check(variable):
    """
    This function to explores the variable values at
    predefined quantile intervals.
    This is will aid in determining capping limits.
    """
    quantile_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    for quantile in quantile_range:
        interval = variable.quantile(quantile)
        print(f"The '{quantile}' quantile value is '{interval}'.")
```


```python
lower_quantile_check(data.MonthlyIncome)
```

    The '0' quantile value is '1000.0'.
    The '0.05' quantile value is '17311.7'.
    The '0.1' quantile value is '17686.0'.
    The '0.15' quantile value is '18291.149999999998'.
    The '0.2' quantile value is '19821.0'.
    The '0.25' quantile value is '20485.0'.
    

There is a rapid jump between the 0 and 5 percentile.
Let's explore these rows


```python
data[data.MonthlyIncome<data.MonthlyIncome.quantile(0.05)]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>2</th>
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
      <th>14</th>
      <td>1</td>
      <td>28.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>30.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>2</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Single</td>
      <td>6.0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>Executive</td>
      <td>17028.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>21.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>21.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>3</td>
      <td>3.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Single</td>
      <td>2.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>16232.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>30.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>15.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>2</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Single</td>
      <td>2.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>17206.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1</td>
      <td>39.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>11.0</td>
      <td>Large Business</td>
      <td>Male</td>
      <td>2</td>
      <td>3.0</td>
      <td>Super Deluxe</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>4.0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1.0</td>
      <td>Manager</td>
      <td>17086.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2403</th>
      <td>1</td>
      <td>28.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>7.0</td>
      <td>Large Business</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>5.0</td>
      <td>Single</td>
      <td>7.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>17080.0</td>
    </tr>
    <tr>
      <th>2404</th>
      <td>1</td>
      <td>25.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>15.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>2</td>
      <td>3.0</td>
      <td>Multi</td>
      <td>5.0</td>
      <td>Single</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>Executive</td>
      <td>17096.0</td>
    </tr>
    <tr>
      <th>2422</th>
      <td>1</td>
      <td>31.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>19.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>Super Deluxe</td>
      <td>5.0</td>
      <td>Married</td>
      <td>6.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>Manager</td>
      <td>17302.0</td>
    </tr>
    <tr>
      <th>2442</th>
      <td>1</td>
      <td>18.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>15.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>2</td>
      <td>3.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Single</td>
      <td>2.0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>16611.0</td>
    </tr>
    <tr>
      <th>2586</th>
      <td>0</td>
      <td>39.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>10.0</td>
      <td>Large Business</td>
      <td>Female</td>
      <td>3</td>
      <td>4.0</td>
      <td>Super Deluxe</td>
      <td>3.0</td>
      <td>Single</td>
      <td>5.0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>Manager</td>
      <td>4678.0</td>
    </tr>
  </tbody>
</table>
<p>245 rows × 19 columns</p>
</div>



Observation:
* There are many rows where the `MonthlyIncome` is less than the 5 percentile but there some rows where the monthly income is less than 15000.


```python
# Check for rows where `MonthlyIncome` is less than 15000
data[data.MonthlyIncome<15000]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>142</th>
      <td>0</td>
      <td>38.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>9.0</td>
      <td>Large Business</td>
      <td>Female</td>
      <td>2</td>
      <td>3.0</td>
      <td>Super Deluxe</td>
      <td>3.0</td>
      <td>Single</td>
      <td>4.0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>Manager</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>2586</th>
      <td>0</td>
      <td>39.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>10.0</td>
      <td>Large Business</td>
      <td>Female</td>
      <td>3</td>
      <td>4.0</td>
      <td>Super Deluxe</td>
      <td>3.0</td>
      <td>Single</td>
      <td>5.0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>Manager</td>
      <td>4678.0</td>
    </tr>
  </tbody>
</table>
</div>



There are 2 Female customers, who are similiar in age, 'Manager', 'Single' & have 'MonthlyIncome' less than 15000.  
Let's cap the lowest 'MonthlyIncome' values at 15000 as the presence of such small outliers will skew the data.


```python
# capping outliers ('MonthlyIncome' less than 15000) at 15000

data.MonthlyIncome = np.where(data.MonthlyIncome < 15000, 15000,
                              data.MonthlyIncome)
```

---

**Upper Outliers**


```python
quantile_check(data.MonthlyIncome)
```

    The '0.8' quantile value is '26867.0'.
    The '0.9' quantile value is '31869.9'.
    The '0.95' quantile value is '34632.85'.
    The '0.98' quantile value is '37418.0'.
    The '0.99' quantile value is '38084.0'.
    The '0.995' quantile value is '38310.08499999999'.
    The '1.0' quantile value is '98678.0'.
    

There is a rapid jump after the 99.5 percentile.
Let's explore these rows


```python
data[data.MonthlyIncome>data.MonthlyIncome.quantile(0.995)]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>38</th>
      <td>0</td>
      <td>36.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>11.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>2</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>Executive</td>
      <td>95000.0</td>
    </tr>
    <tr>
      <th>2482</th>
      <td>0</td>
      <td>37.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>12.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>3</td>
      <td>5.0</td>
      <td>Multi</td>
      <td>5.0</td>
      <td>Divorced</td>
      <td>2.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>98678.0</td>
    </tr>
    <tr>
      <th>2609</th>
      <td>0</td>
      <td>51.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>18.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>3</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Single</td>
      <td>5.0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
      <td>VP</td>
      <td>38604.0</td>
    </tr>
    <tr>
      <th>2634</th>
      <td>0</td>
      <td>53.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>7.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>5.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2.0</td>
      <td>VP</td>
      <td>38677.0</td>
    </tr>
    <tr>
      <th>3012</th>
      <td>1</td>
      <td>56.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>9.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>4</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>7.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3.0</td>
      <td>VP</td>
      <td>38537.0</td>
    </tr>
    <tr>
      <th>3190</th>
      <td>0</td>
      <td>42.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>14.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>3</td>
      <td>6.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1.0</td>
      <td>VP</td>
      <td>38651.0</td>
    </tr>
    <tr>
      <th>3193</th>
      <td>1</td>
      <td>53.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>9.0</td>
      <td>Small Business</td>
      <td>Female</td>
      <td>3</td>
      <td>6.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>3.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>VP</td>
      <td>38523.0</td>
    </tr>
    <tr>
      <th>3295</th>
      <td>0</td>
      <td>57.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>11.0</td>
      <td>Large Business</td>
      <td>Female</td>
      <td>4</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>6.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3.0</td>
      <td>VP</td>
      <td>38621.0</td>
    </tr>
    <tr>
      <th>3342</th>
      <td>0</td>
      <td>44.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>10.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>6.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>5.0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>3.0</td>
      <td>VP</td>
      <td>38418.0</td>
    </tr>
    <tr>
      <th>3362</th>
      <td>0</td>
      <td>52.0</td>
      <td>Company Invited</td>
      <td>3</td>
      <td>16.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>6.0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2.0</td>
      <td>VP</td>
      <td>38525.0</td>
    </tr>
    <tr>
      <th>3400</th>
      <td>0</td>
      <td>57.0</td>
      <td>Self Enquiry</td>
      <td>2</td>
      <td>15.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Single</td>
      <td>8.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1.0</td>
      <td>VP</td>
      <td>38395.0</td>
    </tr>
    <tr>
      <th>3453</th>
      <td>0</td>
      <td>59.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>7.0</td>
      <td>Small Business</td>
      <td>Female</td>
      <td>4</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>5.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>VP</td>
      <td>38379.0</td>
    </tr>
    <tr>
      <th>3598</th>
      <td>0</td>
      <td>48.0</td>
      <td>Self Enquiry</td>
      <td>2</td>
      <td>33.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>4</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>5.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>2.0</td>
      <td>VP</td>
      <td>38336.0</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>0</td>
      <td>41.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>14.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Single</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>2.0</td>
      <td>VP</td>
      <td>38511.0</td>
    </tr>
    <tr>
      <th>3775</th>
      <td>0</td>
      <td>49.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>17.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>5.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>6.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2.0</td>
      <td>VP</td>
      <td>38343.0</td>
    </tr>
    <tr>
      <th>3845</th>
      <td>0</td>
      <td>56.0</td>
      <td>Self Enquiry</td>
      <td>2</td>
      <td>33.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>2.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>6.0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>3.0</td>
      <td>VP</td>
      <td>38314.0</td>
    </tr>
    <tr>
      <th>4079</th>
      <td>0</td>
      <td>51.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>18.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>3</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Single</td>
      <td>5.0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>2.0</td>
      <td>VP</td>
      <td>38604.0</td>
    </tr>
    <tr>
      <th>4104</th>
      <td>0</td>
      <td>53.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>7.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>5.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>VP</td>
      <td>38677.0</td>
    </tr>
    <tr>
      <th>4482</th>
      <td>1</td>
      <td>56.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>9.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>4</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>7.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>VP</td>
      <td>38537.0</td>
    </tr>
    <tr>
      <th>4660</th>
      <td>0</td>
      <td>42.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>14.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>3</td>
      <td>6.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>2.0</td>
      <td>VP</td>
      <td>38651.0</td>
    </tr>
    <tr>
      <th>4663</th>
      <td>1</td>
      <td>53.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>9.0</td>
      <td>Small Business</td>
      <td>Female</td>
      <td>3</td>
      <td>6.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2.0</td>
      <td>VP</td>
      <td>38523.0</td>
    </tr>
    <tr>
      <th>4765</th>
      <td>0</td>
      <td>57.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>11.0</td>
      <td>Large Business</td>
      <td>Female</td>
      <td>4</td>
      <td>4.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>6.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>2.0</td>
      <td>VP</td>
      <td>38621.0</td>
    </tr>
    <tr>
      <th>4812</th>
      <td>0</td>
      <td>44.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>10.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>6.0</td>
      <td>King</td>
      <td>3.0</td>
      <td>Married</td>
      <td>5.0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
      <td>VP</td>
      <td>38418.0</td>
    </tr>
    <tr>
      <th>4832</th>
      <td>1</td>
      <td>52.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>35.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>4</td>
      <td>5.0</td>
      <td>Super Deluxe</td>
      <td>3.0</td>
      <td>Single</td>
      <td>5.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1.0</td>
      <td>Manager</td>
      <td>38525.0</td>
    </tr>
    <tr>
      <th>4870</th>
      <td>1</td>
      <td>57.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>23.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>4</td>
      <td>4.0</td>
      <td>Standard</td>
      <td>3.0</td>
      <td>Single</td>
      <td>4.0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>3.0</td>
      <td>Senior Manager</td>
      <td>38395.0</td>
    </tr>
  </tbody>
</table>
</div>



Observation:
* There are many rows where the `MonthlyIncome` is greater than the 99.5 percentile but there some rows where the monthly income is greater than 40000.


```python
# Check for rows where `MonthlyIncome` is greater than 40000
data[data.MonthlyIncome>40000]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>38</th>
      <td>0</td>
      <td>36.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>11.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>2</td>
      <td>4.0</td>
      <td>Multi</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>Executive</td>
      <td>95000.0</td>
    </tr>
    <tr>
      <th>2482</th>
      <td>0</td>
      <td>37.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>12.0</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>3</td>
      <td>5.0</td>
      <td>Multi</td>
      <td>5.0</td>
      <td>Divorced</td>
      <td>2.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>98678.0</td>
    </tr>
  </tbody>
</table>
</div>



There are 2 Female customers, who are similiar in age, 'Executive', 'Divorced' & have 'MonthlyIncome' greater than 90000.  
Let's cap the upper 'MonthlyIncome' values at 40000 as the presence of such large outliers will skew the data.


```python
# capping outliers ('MonthlyIncome' greater than 90000) at 40000

data.MonthlyIncome = np.where(data.MonthlyIncome > 40000, 40000,
                              data.MonthlyIncome)
```

---
