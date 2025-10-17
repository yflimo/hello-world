# Problem Statement

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

