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

```python
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters
    Returns:
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """
    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db
```

How gradient descent utilizes the partial derivative of the cost with respect to a parameter at a point to update that parameter.Let's use our `compute_gradient` function to find and plot some partial derivatives of our cost function relative to one of the parameters  w0.

```python
plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()
```

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/c97d019e-65e7-4cff-ae59-30a06ee43904" />

Above, the left plot shows $\frac{\partial J(w,b)}{\partial w}$ or the slope of the cost curve relative to w at three points. On the right side of the plot, the derivative is positive, while on the left it is negative. Due to the 'bowl shape', the derivatives will always lead gradient descent toward the bottom where the gradient is zero.

The left plot has fixed b = 100. Gradient descent will utilize both $\frac{\partial J(w,b)}{\partial w}$ and $\frac{\partial J(w,b)}{\partial b}$ to update parameters. The 'quiver plot' on the right provides a means of viewing the gradient of both parameters. The arrow sizes reflect the magnitude of the gradient at that point. The direction and slope of the arrow reflects the ratio of $\frac{\partial J(w,b)}{\partial w}$ and $\frac{\partial J(w,b)}{\partial b}$ at that point. Note that the gradient points *away* from the minimum. Review equation (3) above. The scaled gradient is *subtracted* from the current value of w or b . This moves the parameter in a direction that will reduce cost.

## Gradient Descent

Now that gradients can be computed, gradient descent, described in equation (3) above can be implemented below in `gradient_descent`. The details of the implementation are described in the comments. Below, you will utilize this function to find optimal values of w and b on the training data.
