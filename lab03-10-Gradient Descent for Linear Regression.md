## Goals
In this lab:
- automate the process of optimizing **w** and **b** using gradient descent.

## Tools
In this lab,will make use of:
- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data
- plotting routines in the `lab_utils.py` file in the local directory

```python
import math, copy
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
```

## Problem Statement
Let’s use the same two data points as before - a house with 1000 square feet sold for $300,000 and a house with 2000 square feet sold for $500,000.

| Size (1000 sqft) | Price (1000s of dollars) |
|------------------|---------------------------|
| 1                | 300                       |
| 2                | 500                       |

```python
x_train = np.array([1.0, 2.0])      #features
y_train = np.array([300.0, 500.0])  #target value
```

## Compute_Cost
This was developed in the last lab. We'll need it again here.

```python
#Function to calculate the cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost
    return total_cost
```

## Gradient descent summary
Developed a linear model that predicts $f_{w,b}(x^{(i)})$：

$$ f_{w,b}(x^{(i)}) = wx^{(i)} + b $$

In linear regression, utilize input training data to fit the parameters w,b by minimizing a measure of the error between our predictions $f_{w,b}(x^{(i)})$ and the actual data y^{(i)}. The measure is called the cost, J(w,b). In training you measure the cost over all of our training samples ( x^{(i)}, y^{(i)} )

- **compute_cost：** 

$$ J(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

In lecture, gradient descent was described as:

repeat until convergence{

$$ w = w - \alpha \frac{\partial J(w,b)}{\partial w} $$

$$ b = b - \alpha \frac{\partial J(w,b)}{\partial b} $$

}

where, parameters w, b are updated simultaneously.

The gradient is defined as:

- **compute_gradient：** 

$$ \frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} $$

$$ \frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) $$

Here simultaneously means that you calculate the partial derivatives for all the parameters before updating any of the parameters.

# Implement Gradient Descent

You will implement gradient descent algorithm for one feature. You will need three functions.

- **compute_gradient** 
- **compute_cost** 
- **gradient_descent**, utilizing **compute_gradient** and **compute_cost**

### Conventions:
- The naming of python variables containing partial derivatives follows this pattern, $\frac{\partial J(w,b)}{\partial b}$ will be `dj_db`.
  
包含偏导数的 Python 变量命名遵循以下模式, $\frac{\partial J(w,b)}{\partial b}$ 会被命名为 `dj_db`.

- w.r.t is With Respect To, as in partial derivative of $J(wb)$ With Respect To $b$.

w.r.t 是 “关于” 的意思，比如 $J(wb)$关于 b 的偏导数

## compute_gradient
`compute_gradient` implements expressions above and returns $\frac{\partial J(w,b)}{\partial w}$, $\frac{\partial J(w,b)}{\partial b}$. The embedded comments describe the operations.

