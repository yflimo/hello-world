## Tools
- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data
- local plotting routines in the `lab_utils_uni.py` file in the local directory

```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')
```

## Problem Statement

| Size (1000 sqft) | Price (1000s of dollars) |
|------------------|--------------------------|
| 1                | 300                      |
| 2                | 500                      |

```python
# In [3]:
x_train = np.array([1.0, 2.0])  # (size in 1000 square feet)
y_train = np.array([300.0, 500.0])  # (price in 1000s of dollars)
```
# Computing Cost

'cost' is a measure how well model is predicting the target price of the house.

The equation for cost with one variable is:

<img width="1384" height="358" alt="image" src="https://github.com/user-attachments/assets/6f9f2ab0-6f33-46d8-a7eb-54a62389faaa" />

The code below calculates cost by looping over each example. 
In each loop:
- `f_wb`, a prediction is calculated
- the difference between the target and the prediction is calculated and squared.
- this is added to the total cost.

```python
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters

    Returns:
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost
```
# Cost  Function Intuition

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/b2a21074-80ac-4dcb-80c9-390c9d6e10c3" />

**goal：** To find a model \( f_{w,b}(x) = wx + b \), with parameters w, b , which will accurately predict house values given an input x.
The cost is a measure of how accurate the model is on the training data.

The cost equation(1) shows that if  w  and  b  can be selected such that the predictions \( f_{w,b}(x) \) match the target data  y , the \( (f_{w,b}(x^{(i)}) - y^{(i)})^2 \) term will be zero and the cost minimized. In this simple two-point example, you can achieve this!

In the previous lab,  determined that  b = 100 provided an optimal solution so let’s set b to 100 and focus on  w.

Below, use the slider control to select the value of w that minimizes cost. It can take a few seconds for the plot to update.

```python
plt_intuition(x_train, y_train)
```

- Cost is minimized when w = 200 , which matches results from the previous lab.
- Because the difference between the target and prediction is squared in the cost equation, the cost increases rapidly when  w is either too large or too small.
- Using the  w  and b selected by minimizing cost results in a line which is a perfect fit to the data.

