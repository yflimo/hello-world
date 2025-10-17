# Problem Statement

<img width="488" height="293" alt="image" src="https://github.com/user-attachments/assets/272c641d-3079-42c9-b78a-5f8d4d614626" />

 use the motivating example of housing price prediction.

This lab will use a simple data set with only two data points:

  1)a house with 1000 square feet(sqft) sold for $300,000

  2)a house with 2000 square feet sold for $500,000

These two points will constitute data or training set. 


| Size (1000 sqft) | Price (1000s of dollars) |
|------------------|--------------------------|
| 1.0              | 300                      |
| 2.0              | 500                      |

fit a linear regression model through these two points


create `x_train` and `y_train` variables. The data is stored in one-dimensional NumPy arrays.

```python
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
```

`x_train = [1. 2.]`

`y_train = [300. 500.]`


## Number of training examples `m` 训练样本数量m

use `m` to denote the number of training examples. 

Numpy arrays have a `.shape` parameter. `x_train.shape` returns a python tuple with an entry for each dimension. 

`x_train.shape[0]` is the length of the array and number of examples

```python
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
```

Output:
```
x_train.shape: (2,)
Number of training examples is: 2
```

One can also use the Python `len()` function as shown below.

```python
# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")
```

Output:
```
Number of training examples is: 2
```

## Training example `x_i`, `y_i` 训练样本 `x_i`, `y_i`

use $(x^{(i)}, y^{(i)})$ to denote the $i^{th}$ training example. 

Since Python is zero indexed, $(x^{(0)}, y^{(0)})$ is (1.0, 300.0) and $(x^{(1)}, y^{(1)})$ is (2.0, 500.0).   Python 是从 0 开始索引的

To access a value in a Numpy array, one indexes the array with the desired offset. 

For example the syntax to access location zero of `x_train` is `x_train[0]`.

Run the next code block below to get the $i^{th}$ training example.

```python
i = 0 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
```

Output:
```
(x^(0), y^(0)) = (1.0, 300.0)
```

## Plotting the data  绘制数据

plot these two points using the `scatter()` function in the `matplotlib` library


- The function arguments `marker` and `c` show the points as red crosses红色叉号 (the default is blue dots默认为蓝色圆点).

use other functions in the `matplotlib` library to set the title标题 and labels标签 to display

```python
# Plot the data points 绘制数据点
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title 设置标题
plt.title("Housing Prices")
# Set the y-axis label 设置y轴标签
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label 设置x轴标签
plt.xlabel('Size (1000 sqft)')
plt.show()
```
<img width="497" height="337" alt="image" src="https://github.com/user-attachments/assets/6da54b5d-dd46-4ec2-89f2-cfcc923f53ff" />


# Model function   函数模型
<img width="531" height="324" alt="image" src="https://github.com/user-attachments/assets/9ff1a361-ef0f-4c15-aa55-48f8f7b515c1" />

the model function for linear regression (which is a function that maps from x to y )       x -> y

<img width="219" height="44" alt="619c6b2111a0dab665eeefd430c913ef" src="https://github.com/user-attachments/assets/14c1e952-2fd1-4f27-a2b3-999e81ece683" />

*Note: come back to this cell to adjust the model's w and b  parameters*  

```python
w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")
```

Output:  
```
w: 100
b: 100
```  

compute the value of f(x) for your two data points. explicitly write this out for each data point as

<img width="496" height="164" alt="image" src="https://github.com/user-attachments/assets/b24dcbe0-1a30-4625-aaf1-507887a98692" />

For a large number of data points, this can get unwieldy and repetitive. So, calculate the function output in a `for` loop.

*Note: The argument description `(ndarray (m,))` describes a Numpy n-dimensional array of shape `(m,)`表示形状为 (m,)的Numpy多维数组.
`(scalar)` describes an argument without dimensions, just a magnitude没有维度的参数，只是一个数值大小.*  

*Note: `np.zeros(n)` will return a one-dimensional numpy array with \( n \) entries返回一个具有n个元素的一维 NumPy 数组。*  

```python
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model    计算线性模型的预测值
    Args参数:
        x (ndarray (m,)): Data, m examples   数据包括m个样本
        w,b (scalar)    : model parameters   模型参数
    Returns
        y (ndarray (m,)): target values      目标值
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb
```

call the `compute_model_output` function and plot the output.   绘图

```python
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot model prediction   绘制模型预测
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

# Plot the data points  绘制数据点
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# Set the title  设置标题
plt.title("Housing Prices")

# Set the y-axis label    设置y轴标签
plt.ylabel('Price (in 1000s of dollars)')

# Set the x-axis label    设置x轴标签
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```

调用 compute_model_output 函数，

传入训练数据 x_train 以及模型参数 w 和 b 得到预测值 tmp_f_wb。

通过 matplotlib 库用蓝色直线绘制模型的预测结果，用红色叉号绘制实际的数据点

设置图表的标题、坐标轴标签和图例

展示图表，以此来对比线性模型对房价（基于房屋面积）的预测情况与实际数据。
<img width="555" height="366" alt="image" src="https://github.com/user-attachments/assets/a260ccaa-16e7-430a-b8e2-5efe409cf961" />

setting w = 100 and b = 100 does not result in a line that fits datas.

# Prediction  预测

Now that we have a model, we can use it to make our original prediction. 

Let's predict the price of a house with 1200 sqft. Since the units of x are in 1000's of sqft, x  is 1.2.

```python
w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")
```

