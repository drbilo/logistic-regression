# Logistic Regression

## Introduction
The last [two](https://github.com/drbilo/linear_regression) [posts](https://github.com/drbilo/multivariate-linear-regression) delt with linear regression where we train a model to output a value given either a single or multiple input variables. Now we will move into the field of classification with Logistic Regression. With this we attempt to predict which class something belongs to given some form of input variables. For example, we would like to predict if an email is spam or not spam or if a transaction is fraudulent or genuine.

### Binary Classification

For the example of 'spam' or 'not spam', we instead would want an output number of either 0, or 1 as our output. Technically, 0 stands for 'negative class' and 1 stands for 'positive class' but in this case 0 would be 'not spam' and 1 would be 'spam'.

### Logistic Function

Previously we used the simple ```y=mx + b``` to guess our value but since classification isn't a linear problem we will instead have to use a different hypothesis function. 

![alt text](https://www.dropbox.com/s/u91yq42uegnz46x/logistic%20function.png?raw=1 "logistic function")

The benefit of using this function is that it gives us a 0 or 1 value given any input number and is thus better suited for classification.

![alt text](https://www.dropbox.com/s/z9pkvcyfd5o4epg/logistic%20curve.png?raw=1 "logistic curve")

The implementation in the code for this functions is:

```Python
def sigmoid(z):
    return  1 / (1 + np.exp(-z))
```

### Cost Function

Like with our hypothesis function, we cannot use a the same cost function as we use in linear regression as this would result in a wavy line with many local optima so gradient descent wouldn't find the global minimum value. To solve this, we have to use a modified cost function that gives us a convex function so we can find optimal values. The updated cost function:

![alt text](https://www.dropbox.com/s/gagmn7ocgpjblqk/cost%20function.png?raw=1 "cost function")

Or a vectorized version:

![alt text](https://www.dropbox.com/s/opojq3mnbtvgiov/vector%20cost%20function.png?raw=1 "vectorized cost function")

This is implemented in our code as:

```python
def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
```

### Gradient Descent

To minimize our cost function we can use the exact same gradient descent algorithm used in linear regression.

```python
def gradient(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

def logistic_regression(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    for i in range(iters):
        h = sigmoid(np.dot(X, theta))
        cost_num = cost(h, y)
        cost_array[i] = cost_num
        gradient_val = gradient(X, h, y)
        theta = theta - (gradient_val * alpha)
    return theta, cost_array
```

## The Data

To test my implementation of logistic regression I have decided to use the famous [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). For this example I have only decided to use 2 features (length and width) and 2 classes (iris setosa = 0 and iris versicolour = 1).

```
   length  width  Type
0     5.1    3.5     0
1     4.9    3.0     0
2     4.7    3.2     0
3     4.6    3.1     0
4     5.0    3.6     0
...
    length  width  Type
95     5.7    3.0     1
96     5.7    2.9     1
97     6.2    2.9     1
98     5.1    2.5     1
99     5.7    2.8     1
```

Shown on a scatter plot:
![alt text](https://www.dropbox.com/s/exo1yokxm2e0v9d/scatteriris.png?raw=1 "scatter data")

## Results

With starting theta values of 0 our cost function gives us a value of: ```Initial cost value for theta values [0. 0. 0.] is: 0.6931471805599453```

With a learning rate of 0.01 and iterations of 10,000, our logistic regression algorithm gives the following result: ```Final cost value for theta values [-0.70846899  3.04714971 -5.10943662] is: 0.10131731164299049```

![alt text](https://www.dropbox.com/s/pq8zqk88uhw4o44/erroriterations.png?raw=1 "error iterations chat")

```
Hypothesis = 0.05 actual =  0 
Hypothesis = 0.25 actual =  0 
Hypothesis = 0.06 actual =  0 
Hypothesis = 0.07 actual =  0 
Hypothesis = 0.02 actual =  0 
Hypothesis = 0.02 actual =  0 
Hypothesis = 0.02 actual =  0 
Hypothesis = 0.05 actual =  0 
Hypothesis = 0.11 actual =  0 
Hypothesis = 0.17 actual =  0 
Hypothesis = 0.04 actual =  0 
Hypothesis = 0.03 actual =  0 
Hypothesis = 0.20 actual =  0 
Hypothesis = 0.05 actual =  0 
Hypothesis = 0.03 actual =  0 
Hypothesis = 0.00 actual =  0 
Hypothesis = 0.02 actual =  0 
Hypothesis = 0.05 actual =  0 
Hypothesis = 0.06 actual =  0 
Hypothesis = 0.01 actual =  0 
Hypothesis = 0.16 actual =  0 
Hypothesis = 0.02 actual =  0 
Hypothesis = 0.01 actual =  0 
Hypothesis = 0.12 actual =  0 
Hypothesis = 0.03 actual =  0 
Hypothesis = 0.31 actual =  0 
Hypothesis = 0.05 actual =  0 
Hypothesis = 0.06 actual =  0 
Hypothesis = 0.10 actual =  0 
Hypothesis = 0.06 actual =  0 
Hypothesis = 0.13 actual =  0 
Hypothesis = 0.16 actual =  0 
Hypothesis = 0.00 actual =  0 
Hypothesis = 0.00 actual =  0 
Hypothesis = 0.17 actual =  0 
Hypothesis = 0.14 actual =  0 
Hypothesis = 0.14 actual =  0 
Hypothesis = 0.17 actual =  0 
Hypothesis = 0.07 actual =  0 
Hypothesis = 0.07 actual =  0 
Hypothesis = 0.03 actual =  0 
Hypothesis = 0.78 actual =  0 
Hypothesis = 0.03 actual =  0 
Hypothesis = 0.03 actual =  0 
Hypothesis = 0.01 actual =  0 
Hypothesis = 0.20 actual =  0 
Hypothesis = 0.01 actual =  0 
Hypothesis = 0.05 actual =  0 
Hypothesis = 0.03 actual =  0 
Hypothesis = 0.09 actual =  0 
Hypothesis = 0.99 actual =  1 
Hypothesis = 0.92 actual =  1 
Hypothesis = 0.99 actual =  1 
Hypothesis = 0.99 actual =  1 
Hypothesis = 0.99 actual =  1 
Hypothesis = 0.91 actual =  1 
Hypothesis = 0.84 actual =  1 
Hypothesis = 0.88 actual =  1 
Hypothesis = 0.99 actual =  1 
Hypothesis = 0.79 actual =  1 
Hypothesis = 0.99 actual =  1 
Hypothesis = 0.87 actual =  1 
Hypothesis = 1.00 actual =  1 
Hypothesis = 0.96 actual =  1 
Hypothesis = 0.82 actual =  1 
Hypothesis = 0.98 actual =  1 
Hypothesis = 0.74 actual =  1 
Hypothesis = 0.96 actual =  1 
Hypothesis = 1.00 actual =  1 
Hypothesis = 0.97 actual =  1 
Hypothesis = 0.71 actual =  1 
Hypothesis = 0.97 actual =  1 
Hypothesis = 1.00 actual =  1 
Hypothesis = 0.97 actual =  1 
Hypothesis = 0.98 actual =  1 
Hypothesis = 0.98 actual =  1 
Hypothesis = 1.00 actual =  1 
Hypothesis = 0.99 actual =  1 
Hypothesis = 0.94 actual =  1 
Hypothesis = 0.97 actual =  1 
Hypothesis = 0.98 actual =  1 
Hypothesis = 0.98 actual =  1 
Hypothesis = 0.96 actual =  1 
Hypothesis = 0.98 actual =  1 
Hypothesis = 0.60 actual =  1 
Hypothesis = 0.55 actual =  1 
Hypothesis = 0.98 actual =  1 
Hypothesis = 1.00 actual =  1 
Hypothesis = 0.74 actual =  1 
Hypothesis = 0.96 actual =  1 
Hypothesis = 0.94 actual =  1 
Hypothesis = 0.93 actual =  1 
Hypothesis = 0.98 actual =  1 
Hypothesis = 0.94 actual =  1 
Hypothesis = 0.93 actual =  1 
Hypothesis = 0.79 actual =  1 
Hypothesis = 0.86 actual =  1 
Hypothesis = 0.97 actual =  1 
Hypothesis = 0.89 actual =  1 
Hypothesis = 0.91 actual =  1
```

Given ```hθ(x) ≥0.5 → y = 1 hθ(x) <0.5 → y = 0``` then our model has been trained succesfully.

### Usage

`python irislogisticreg.py`

## Links
* [Logistic Regression from scratch in Python](https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac)

* [ML: Logistic Regression](https://www.coursera.org/learn/machine-learning/)