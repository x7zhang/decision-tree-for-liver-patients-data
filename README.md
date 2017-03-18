
# Decision Tree Model for Liver Patients Data

### Introduction

Patients Suffering from Liver disease have been continuously increasing because of excessive consumption of alcohol, inhale of harmful gases, intake of contaminated food and drugs.
Problems with liver patients are not easily discovered in an early stage, as the liver will be functioning normally even when it is partially damaged.
Liver disease can be diagnosed by analyzing the levels of enzymes in the blood. An early diagnosis of liver problems will help to increase the patient’ s survival rate.


```python
#Import Libraries
import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
import sys
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'
```


```python
liver_patients = pd.read_csv("sample_liver_patients.csv")# Training dataset
liver_patients.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>TB</th>
      <th>DB</th>
      <th>Alkphos</th>
      <th>Sgpt</th>
      <th>Sgot</th>
      <th>TP</th>
      <th>ALB</th>
      <th>AGratio</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65</td>
      <td>Female</td>
      <td>0.7</td>
      <td>0.1</td>
      <td>187</td>
      <td>16</td>
      <td>18</td>
      <td>6.8</td>
      <td>3.3</td>
      <td>0.90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>62</td>
      <td>Male</td>
      <td>10.9</td>
      <td>5.5</td>
      <td>699</td>
      <td>64</td>
      <td>100</td>
      <td>7.5</td>
      <td>3.2</td>
      <td>0.74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62</td>
      <td>Male</td>
      <td>7.3</td>
      <td>4.1</td>
      <td>490</td>
      <td>60</td>
      <td>68</td>
      <td>7.0</td>
      <td>3.3</td>
      <td>0.89</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58</td>
      <td>Male</td>
      <td>1.0</td>
      <td>0.4</td>
      <td>182</td>
      <td>14</td>
      <td>20</td>
      <td>6.8</td>
      <td>3.4</td>
      <td>1.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>72</td>
      <td>Male</td>
      <td>3.9</td>
      <td>2.0</td>
      <td>195</td>
      <td>27</td>
      <td>59</td>
      <td>7.3</td>
      <td>2.4</td>
      <td>0.40</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
liver_patients.isnull().any() #These is null value in column "AGratio"
```




    Age        False
    Gender     False
    TB         False
    DB         False
    Alkphos    False
    Sgpt       False
    Sgot       False
    TP         False
    ALB        False
    AGratio     True
    Target     False
    dtype: bool




```python
liver_patients=liver_patients.fillna(0) #Using "0" to instead of null value
```


```python
liver_patients.isnull().any()
```




    Age        False
    Gender     False
    TB         False
    DB         False
    Alkphos    False
    Sgpt       False
    Sgot       False
    TP         False
    ALB        False
    AGratio    False
    Target     False
    dtype: bool




```python
print("Target:", liver_patients["Target"].unique(), sep="\n")
# 1 means "liver patient", 0 means "non-liver patients"
```

    Target:
    [1 2]
    

## Preprocessing


```python
#Encode the "Gender" to Integer
def encode_target(df, target_column):
    df_mode = df.copy()
    targets = df_mode[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mode["Gender"] = df_mode[target_column].replace(map_to_int)
    
    return(df_mode, targets)
```


```python
df, targets = encode_target(liver_patients, "Gender")
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>TB</th>
      <th>DB</th>
      <th>Alkphos</th>
      <th>Sgpt</th>
      <th>Sgot</th>
      <th>TP</th>
      <th>ALB</th>
      <th>AGratio</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65</td>
      <td>0</td>
      <td>0.7</td>
      <td>0.1</td>
      <td>187</td>
      <td>16</td>
      <td>18</td>
      <td>6.8</td>
      <td>3.3</td>
      <td>0.90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>62</td>
      <td>1</td>
      <td>10.9</td>
      <td>5.5</td>
      <td>699</td>
      <td>64</td>
      <td>100</td>
      <td>7.5</td>
      <td>3.2</td>
      <td>0.74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62</td>
      <td>1</td>
      <td>7.3</td>
      <td>4.1</td>
      <td>490</td>
      <td>60</td>
      <td>68</td>
      <td>7.0</td>
      <td>3.3</td>
      <td>0.89</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.4</td>
      <td>182</td>
      <td>14</td>
      <td>20</td>
      <td>6.8</td>
      <td>3.4</td>
      <td>1.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>72</td>
      <td>1</td>
      <td>3.9</td>
      <td>2.0</td>
      <td>195</td>
      <td>27</td>
      <td>59</td>
      <td>7.3</td>
      <td>2.4</td>
      <td>0.40</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
features = list(liver_patients.columns[:10]) #Features selected to predict target
print("features:", features, sep="\n")
```

    features:
    ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'AGratio']
    

## Fitting the decision tree with scikit-learn


```python
y=df["Target"]
x=df[features]
dt=tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(x, y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=20, min_weight_fraction_leaf=0.0,
                presort=False, random_state=99, splitter='best')



## Visualizing the tree


```python
from IPython.display import Image
import pydotplus
```


```python
dot_data = tree.export_graphviz(dt, out_file = None,
                                feature_names = features
                               )
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 

```




![png](https://github.com/x7zhang/decision-tree-for-liver-patients-data/blob/master/graphs/output_18_0.png?raw=true)




```python
prediction_score=dt.score(X=df[features],
         y=df["Target"])
print("The decision tree model on the training data: ", prediction_score )
```

    The decision tree model on the training data:  0.845188284519
    

### The model is almost 85% accurate on the training data, but how does it on testing data?


```python
test_liver = pd.read_csv("test_liver_patient.csv")
test_liver.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>TB</th>
      <th>DB</th>
      <th>Alkphos</th>
      <th>Sgpt</th>
      <th>Sgot</th>
      <th>TP</th>
      <th>ALB</th>
      <th>AGratio</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>Male</td>
      <td>7.1</td>
      <td>3.7</td>
      <td>196</td>
      <td>622</td>
      <td>497</td>
      <td>6.9</td>
      <td>3.6</td>
      <td>1.09</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>Male</td>
      <td>3.4</td>
      <td>1.6</td>
      <td>186</td>
      <td>779</td>
      <td>844</td>
      <td>7.3</td>
      <td>3.2</td>
      <td>0.70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>Male</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>352</td>
      <td>28</td>
      <td>51</td>
      <td>7.9</td>
      <td>4.2</td>
      <td>1.10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>Male</td>
      <td>2.3</td>
      <td>1.3</td>
      <td>282</td>
      <td>132</td>
      <td>368</td>
      <td>7.3</td>
      <td>4.0</td>
      <td>1.20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>Male</td>
      <td>1.1</td>
      <td>0.4</td>
      <td>92</td>
      <td>91</td>
      <td>188</td>
      <td>7.2</td>
      <td>3.8</td>
      <td>1.11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df, targets = encode_target(test_liver, "Gender")
```


```python
test_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>TB</th>
      <th>DB</th>
      <th>Alkphos</th>
      <th>Sgpt</th>
      <th>Sgot</th>
      <th>TP</th>
      <th>ALB</th>
      <th>AGratio</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>0</td>
      <td>7.1</td>
      <td>3.7</td>
      <td>196</td>
      <td>622</td>
      <td>497</td>
      <td>6.9</td>
      <td>3.6</td>
      <td>1.09</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>0</td>
      <td>3.4</td>
      <td>1.6</td>
      <td>186</td>
      <td>779</td>
      <td>844</td>
      <td>7.3</td>
      <td>3.2</td>
      <td>0.70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>352</td>
      <td>28</td>
      <td>51</td>
      <td>7.9</td>
      <td>4.2</td>
      <td>1.10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>0</td>
      <td>2.3</td>
      <td>1.3</td>
      <td>282</td>
      <td>132</td>
      <td>368</td>
      <td>7.3</td>
      <td>4.0</td>
      <td>1.20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>0</td>
      <td>1.1</td>
      <td>0.4</td>
      <td>92</td>
      <td>91</td>
      <td>188</td>
      <td>7.2</td>
      <td>3.8</td>
      <td>1.11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df = test_df.fillna(0)
```


```python
test_df.isnull().any()
```




    Age        False
    Gender     False
    TB         False
    DB         False
    Alkphos    False
    Sgpt       False
    Sgot       False
    TP         False
    ALB        False
    AGratio    False
    Target     False
    dtype: bool




```python
test_features = list(test_df.columns[:10])
test_features
```




    ['Age',
     'Gender',
     'TB',
     'DB',
     'Alkphos',
     'Sgpt',
     'Sgot',
     'TP',
     'ALB',
     'AGratio']




```python
# Make test set predictions
test_preds = dt.predict(X=test_df[test_features])

# Create a submission
submission = pd.DataFrame({"Origin-Target" : test_df["Target"],
                           "Test-Result" : test_preds})

# Save submission to CSV
submission.to_csv("result.csv", index = False)
```


```python
result = pd.read_csv("result.csv")
result.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Origin-Target</th>
      <th>Test-Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The accurate for model in testing data is almost 77%

## Holdout Validation and Cross Validation


Holdout validation and cross validation are two common methods for assessing a model before using it on test data. Holdout validation involves splitting the training data into two parts, a training set and a validation set, building a model with the training set and then assessing performance with the validation set. In theory, model performance on the hold-out validation set should roughly mirror the performance you'd expect to see on unseen test data. In practice, holdout validation is fast and it can work well, especially on large data sets, but it has some pitfalls.


```python
from sklearn.model_selection import train_test_split
```


```python
liver_all = pd.read_csv("Liver Patient Dataset.csv") 
liver_all, targets= encode_target(liver_all, "Gender")
liver_all=liver_all.fillna(0)
```


```python
liver_all.isnull().any()
```




    Age        False
    Gender     False
    TB         False
    DB         False
    Alkphos    False
    Sgpt       False
    Sgot       False
    TP         False
    ALB        False
    AGratio    False
    Target     False
    dtype: bool




```python
v_train, v_test = train_test_split(liver_all, test_size = 0.25,
                                   random_state=1,
                                   stratify = liver_all["Target"])
```


```python
print(v_train.shape)
print(v_test.shape)
```

    (437, 11)
    (146, 11)
    


```python
from sklearn.cross_validation import KFold

```


```python
cv = KFold(n=len(liver_all),  # Number of elements
           n_folds=10,            # Desired number of cv folds
           random_state=12)       # Set a random seed
```


```python
cv
```




    sklearn.cross_validation.KFold(n=583, n_folds=10, shuffle=False, random_state=12)




```python
fold_accuracy = []

for train_fold, valid_fold in cv:
    train = liver_all.loc[train_fold] # Extract train data with cv indices
    valid = liver_all.loc[valid_fold] # Extract valid data with cv indices
    
    model = dt.fit(X=train[features],
                   y=train["Target"])
    valid_acc = model.score(X=valid[features],
                            y=valid["Target"])
    
    fold_accuracy.append(valid_acc)

print("Accuracy per fold: ", fold_accuracy, "\n")
print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))
```

    Accuracy per fold:  [0.71186440677966101, 0.59322033898305082, 0.82476271186440679, 0.65517241379310343, 0.58620689655172409, 0.58620689655172409, 0.58620689655172409, 0.56896551724137934, 0.74137931034482762, 0.72413793103448276] 
    
    Average accuracy:  0.65791233197
    


```python
from sklearn.cross_validation import cross_val_score

```


```python
scores = cross_val_score(estimator= dt,     # Model to test
                X= liver_all[features],  
                y = liver_all["Target"],      # Target variable
                scoring = "accuracy",               # Scoring metric    
                cv=10)                              # Cross validation folds

print("Accuracy per fold: ")
print(scores)
print("Average accuracy: ", scores.mean())
```

    Accuracy per fold: 
    [ 0.71186441  0.61016949  0.72881356  0.62711864  0.54237288  0.69491525
      0.65517241  0.57894737  0.75438596  0.68421053]
    Average accuracy:  0.658797051073
    


```python

```
