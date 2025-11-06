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
#定义成本函数
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

**Goal：** To find a model $$f_{w,b}(x) = wx + b$$ , with parameters w , b, which will accurately predict house values given an input x.
The cost is a measure of how accurate the model is on the training data.

The cost equation shows that if w and b can be selected such that the predictions $$f_{w,b}(x)$$ match the target data y , the $$\left(f_{w,b}\left(x^{(i)}\right) - y^{(i)}\right)^2$$ term will be zero and the cost minimized. 

In the previous lab, determined that b = 100 provided an optimal solution so let’s set b to 100 and focus on w .

Below, use the slider control to select the value of w that minimizes cost. It can take a few seconds for the plot to update.

```python
plt_intuition(x_train, y_train)
```

- Cost is minimized when w = 200 , which matches results from the previous lab.
- Because the difference between the target and prediction is squared in the cost equation, the cost increases rapidly when  w is either too large or too small.
- Using the w and b selected by minimizing cost results in a line which is a perfect fit to the data.

# Cost Function Visualization- 3D

```python
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730,])
```

In the contour plot, click on a point to select \( w \) and \( b \) to achieve the lowest cost. Use the contours to guide your selections. 

```python
plt.close('all')   #关闭当前所有打开的图形窗口
#创建一个交互式图表，可以点击图中不同的 w 和 b，看到预测线和 cost 值如何变化
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()    #显示成本函数的3D形状图。它看起来像个“汤碗”——最低点就是最优解。
```


```python
import numpy as np
%matplotlib notebook
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost
plt_intuition(x_train,y_train)
plt.close('all')
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()
```
