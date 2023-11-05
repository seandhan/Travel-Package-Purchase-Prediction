# Comparing all models


```python
# defining list of models
models = [dtree, dtree_gridCV, rf_estimator, rf_tuned, ab_classifer, ab_tuned,
          gb_estimator, gb_tuned, xgb_estimator, xgb_tuned,
          stacking_estimator]

# defining empty lists to add train and test results
Accuracy_train = []
Accuracy_test = []
Recall_train = []
Recall_test = []

# looping through all the models to get the rmse and r2 scores
for model in models:
    # accuracy score
    j = metrics_summary(model, False)
    Accuracy_train.append(j[0])
    Accuracy_test.append(j[1])
    Recall_train.append(j[2])
    Recall_test.append(j[3])
```


```python
comparison_frame = pd.DataFrame({'Model': ['Decision Tree', 'Tuned Decision Tree', 'Random Forest', 'Tuned Random Forest',
                                           'AdaBoost Regressor', 'Tuned AdaBoost Regressor',
                                           'Gradient Boosting Regressor', 'Tuned Gradient Boosting Regressor',
                                           'XGBoost Regressor',  'Tuned XGBoost Regressor', 'Stacking Regressor'],
                                 'Train_Accuracy': Accuracy_train, 'Test_Accuracy': Accuracy_test,
                                 'Train_Recall': Recall_train, 'Test_Recall': Recall_test})
comparison_frame
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Train_Accuracy</th>
      <th>Test_Accuracy</th>
      <th>Train_Recall</th>
      <th>Test_Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Tree</td>
      <td>0.995323</td>
      <td>0.861622</td>
      <td>0.981366</td>
      <td>0.652174</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tuned Decision Tree</td>
      <td>0.995031</td>
      <td>0.859577</td>
      <td>0.979814</td>
      <td>0.637681</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest</td>
      <td>0.999123</td>
      <td>0.890934</td>
      <td>0.995342</td>
      <td>0.489130</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tuned Random Forest</td>
      <td>0.997077</td>
      <td>0.912065</td>
      <td>0.984472</td>
      <td>0.605072</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AdaBoost Regressor</td>
      <td>0.843905</td>
      <td>0.838446</td>
      <td>0.281056</td>
      <td>0.286232</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tuned AdaBoost Regressor</td>
      <td>0.845659</td>
      <td>0.838446</td>
      <td>0.282609</td>
      <td>0.278986</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gradient Boosting Regressor</td>
      <td>0.878690</td>
      <td>0.862986</td>
      <td>0.428571</td>
      <td>0.362319</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Tuned Gradient Boosting Regressor</td>
      <td>1.000000</td>
      <td>0.921609</td>
      <td>1.000000</td>
      <td>0.655797</td>
    </tr>
    <tr>
      <th>8</th>
      <td>XGBoost Regressor</td>
      <td>0.997369</td>
      <td>0.904567</td>
      <td>0.986025</td>
      <td>0.594203</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Tuned XGBoost Regressor</td>
      <td>0.999415</td>
      <td>0.905930</td>
      <td>0.996894</td>
      <td>0.608696</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Stacking Regressor</td>
      <td>0.997954</td>
      <td>0.912065</td>
      <td>0.990683</td>
      <td>0.673913</td>
    </tr>
  </tbody>
</table>
</div>



---

# Business Recommendations

* This predictive model can be used to determine the customers that the Marketing team should target for the travel agency next package.

* Ideal customers to target would be Executives, having a passport, who are either single or divorced and enquired on travel plans by themselves.

* Having a Passport is the greatest indication that a person will be interested in purchasing a travel product.

* Customers who are not partnered are also those who will have a higher proplensity to travel.

* Free Lancers who travel for their jobs may not be inclined to purchase a travel plan as they may want to settle and relax on their downtime.

---
