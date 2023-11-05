# Data Preparation  

Let's prepare the data for **Bagging** and **Boosting** Ensemble model building.
To do this we must do the following:
1. Drop columns that will add no value to the models
    * Drop the "Customer Interaction Data"
2. Missing value treatment.
3. Outlier treatment.
4. Ensure that the target variable is binary or dichotomous.

---

## Dropping unncessary variables

**Dropping the `Customer Interaction Data` variables**

We shall drop these variables as they are data gathered post pitching which cannot be used to predict any model.


```python
Customer_Interaction_vars = ['PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups','DurationOfPitch']
```


```python
# Drop ZIPCode column inplace
data.drop(columns = Customer_Interaction_vars, inplace=True)
```

For completeness, let's update the `numerical_vars` and `categorical_vars` lists


```python
# Remove 'Customer Interaction variables' in numerical variable list
for variable in Customer_Interaction_vars:
    if variable in numerical_vars:
        numerical_vars.remove(variable)

# Remove 'Customer Interaction variables' in categorical variable list
for variable in Customer_Interaction_vars:
    if variable in categorical_vars:
        categorical_vars.remove(variable)
```

---

## Missing Value Treatment

**Missing Value check**


```python
#Applying the missing value summary function
missing_val_chk(data)
```

    There are NO missing values in the dataset
    

---

## Outlier Treatment


```python
#Applying the Outlier check function for the sub-dataframe of numerical variables
outlier_count(data[numerical_vars])
```

    The 'NumberOfTrips' distribution has '0' lower outliers and '109' upper outliers.
    
    The 'MonthlyIncome' distribution has '0' lower outliers and '373' upper outliers.
    
    

According to literature, **Bagging** Ensemble methods are robust to outliers while **Boosting** Ensemble methods may be sensitive.  
Therefore outlier treatment will only be necessary for the **Boosting** Ensemble methods.

For **Bagging**, we shall use the data without making any changes.

For **Boosting**, we shall clone the data, deal with the outliers and in the end run and compare models with and without outlier treatment.


```python
# Clone data for outlier treatment
data_clean = copy.deepcopy(data)
```

---

### NumberOfTrips


```python
data_clean.NumberOfTrips.describe()
```




    count    4888.000000
    mean        3.219517
    std         1.759505
    min         1.000000
    25%         2.000000
    50%         3.000000
    75%         4.000000
    max         8.000000
    Name: NumberOfTrips, dtype: float64



Between Q3 and Q4, there is significant change compared to (Q1-Q2) & (Q2-Q3).  
Let's explore further to see where there is a rapid jump.


```python
quantile_check(data_clean.NumberOfTrips)
```

    The '0.8' quantile value is '5.0'.
    The '0.9' quantile value is '6.0'.
    The '0.95' quantile value is '7.0'.
    The '0.98' quantile value is '8.0'.
    The '0.99' quantile value is '8.0'.
    The '0.995' quantile value is '8.0'.
    The '1.0' quantile value is '8.0'.
    

After the 95th percentile, the number of trips remain at maximum of 8.  
Let's check how many values greater than 7.0 the series contain.


```python
data_clean[data_clean.NumberOfTrips > data_clean.NumberOfTrips.quantile(
    0.95)]['NumberOfTrips'].value_counts()
```




    8.0    109
    Name: NumberOfTrips, dtype: int64



There are **109** values of 8 which corresponds to the number of outliers.
Therefore let's cap the number of trips at 7.0 (the 95th percentile)


```python
# capping outliers at 95th percentile

data_clean.NumberOfTrips = np.where(
    data_clean.NumberOfTrips > data_clean.NumberOfTrips.quantile(0.95),
    data_clean.NumberOfTrips.quantile(0.95), data_clean.NumberOfTrips)
```

---

### MonthlyIncome


```python
data_clean.MonthlyIncome.describe()
```




    count     4888.000000
    mean     23540.898732
    std       5040.761959
    min      15000.000000
    25%      20485.000000
    50%      22347.000000
    75%      25424.750000
    max      40000.000000
    Name: MonthlyIncome, dtype: float64



Between (Q3 & Q4), there is significant change compared to (Q1 & Q2) & (Q2 & Q3).  
Let's explore further to see where there is a rapid jump.


```python
quantile_check(data_clean.MonthlyIncome)
```

    The '0.8' quantile value is '26867.0'.
    The '0.9' quantile value is '31869.9'.
    The '0.95' quantile value is '34632.85'.
    The '0.98' quantile value is '37418.0'.
    The '0.99' quantile value is '38084.0'.
    The '0.995' quantile value is '38310.08499999999'.
    The '1.0' quantile value is '40000.0'.
    

Since the variable is continuous, let's calculate the upper fence of the box plot that illustrates the outliers


```python
# Calculate the Inter Quartile Range
MonthlyIncome_IQR = spy.stats.iqr(data_clean.MonthlyIncome, nan_policy="omit")
# Calculate the upper fence
upper_fence = data_clean.MonthlyIncome.quantile(0.75) + 1.5 * MonthlyIncome_IQR
upper_fence
```




    32834.375



Let's check the number of rows above the `upper fence`


```python
data_clean[data_clean.MonthlyIncome>upper_fence]
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
      <th>Occupation</th>
      <th>Gender</th>
      <th>NumberOfPersonVisited</th>
      <th>PreferredPropertyStar</th>
      <th>MaritalStatus</th>
      <th>NumberOfTrips</th>
      <th>Passport</th>
      <th>OwnCar</th>
      <th>NumberOfChildrenVisited</th>
      <th>Designation</th>
      <th>MonthlyIncome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>53.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>2</td>
      <td>3.0</td>
      <td>Married</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>VP</td>
      <td>34094.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>46.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>Small Business</td>
      <td>Female</td>
      <td>2</td>
      <td>5.0</td>
      <td>Single</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>VP</td>
      <td>33947.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
      <td>36.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>2</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>Executive</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1</td>
      <td>41.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>Large Business</td>
      <td>Female</td>
      <td>2</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>VP</td>
      <td>34545.0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0</td>
      <td>50.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>Small Business</td>
      <td>Female</td>
      <td>2</td>
      <td>3.0</td>
      <td>Married</td>
      <td>6.0</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>VP</td>
      <td>33740.0</td>
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
    </tr>
    <tr>
      <th>4851</th>
      <td>1</td>
      <td>40.0</td>
      <td>Self Enquiry</td>
      <td>1</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>4</td>
      <td>5.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>35801.0</td>
    </tr>
    <tr>
      <th>4859</th>
      <td>1</td>
      <td>51.0</td>
      <td>Company Invited</td>
      <td>3</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>3</td>
      <td>3.0</td>
      <td>Single</td>
      <td>5.0</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>Manager</td>
      <td>35558.0</td>
    </tr>
    <tr>
      <th>4868</th>
      <td>1</td>
      <td>43.0</td>
      <td>Company Invited</td>
      <td>2</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>4</td>
      <td>3.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>Executive</td>
      <td>36539.0</td>
    </tr>
    <tr>
      <th>4869</th>
      <td>1</td>
      <td>56.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>Small Business</td>
      <td>Female</td>
      <td>3</td>
      <td>4.0</td>
      <td>Single</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>Executive</td>
      <td>37865.0</td>
    </tr>
    <tr>
      <th>4870</th>
      <td>1</td>
      <td>57.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>Salaried</td>
      <td>Female</td>
      <td>4</td>
      <td>3.0</td>
      <td>Single</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>Senior Manager</td>
      <td>38395.0</td>
    </tr>
  </tbody>
</table>
<p>373 rows × 15 columns</p>
</div>



There are **373** values which corresponds to the number of outliers.  
Therefore let's cap the Monthly Income at the upper fence.  


```python
# capping outliers at upper_fence

data_clean.MonthlyIncome = np.where(data_clean.MonthlyIncome > upper_fence,
                                    upper_fence, data_clean.MonthlyIncome)
```

 ---

## Target Variable check

**Binary/ Dichotomoy check**


```python
# Check the unique values
unique_target_states = data.ProdTaken.nunique()
print(f"There are '{unique_target_states}' states of the target variable.")
# Unique value counts
data.ProdTaken.value_counts()
```

    There are '2' states of the target variable.
    




    0    3968
    1     920
    Name: ProdTaken, dtype: int64



<font color='red'>The Target Variable is **Moderately Imbalance**

 ---
