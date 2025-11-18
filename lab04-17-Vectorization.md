# Vectorization
```python
import numpy as np    # it is an unofficial standard to use np for numpy
import time
```
# 1 Goals
- Review the features of NumPy and Python
# 2 Useful References
- NumPy Documentation including a basic introduction:NumPy.org
- A challenging feature topic:NumPy Broadcasting
# 3 Python and NumPy
Python is the programming language we will be using in this course. It has a set of numeric data types and arithmetic operations. 

NumPy is a library that extends the base capabilities of python to add a richer data set including more numeric types, vectors, matrices, and many matrix functions.

NumPy and python work together fairly seamlessly. Python arithmetic operators work on NumPy data types and many NumPy functions will accept python data types.
# 4 Vectors
## 4.1 Abstract
<img width="200" height="150" alt="image" src="https://github.com/user-attachments/assets/9b650d75-9a8e-477d-a287-01b745657307" />

Vectors are ordered arrays of numbers. In notation, vectors are denoted with lower case bold letters such as **x**. 

The elements of a vector are all the same type. A vector does not, for example, contain both characters and numbers. 

**The number of elements in the array** is often referred to as the **dimension** though mathematicians may prefer **rank**. 

The vector shown has a dimension of n. The elements of a vector can be referenced with an index. 

- In math settings, indexes typically run **from 1 to n**. 

- In computer science and these labs, indexing will typically run **from 0 to n-1**.

In notation, elements of a vector, when referenced individually will indicate the index in a subscript, 

for example, the $0^{th}$ element, of the vector x is $x_0$. Note, the x is not bold in this case.
## 4.2 NumPy Arrays
NumPy's basic data structure is an indexable, n-dimensional array containing elements of the same type (dtype). 

可索引的n维数组 包含相同类型的元素

Right away, you may notice we have overloaded the term 'dimension'. Above, it was the number of elements in the vector, here, dimension refers to the number of indexes of an array. A one-dimensional or 1-D array has one index. In Course 1, we will represent vectors as NumPy 1-D arrays.

- 1-D array, shape (n,): n elements indexed [0] through [n-1]

## 4.3NumPy Creation
Data creation routines in NumPy will generally have a first parameter which is the shape of the object. 

This can either be a single value for a 1-D result or a tuple (n,m,...) specifying the shape of the result. 

Below are examples of creating vectors using these routines.
```python
# 默认float
#括号内用int整型表示，生成float（默认 float64） 4个0
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
#括号内用tuple元组表示，生成float（默认 float64） 一维4个0.
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
#生成float（默认 float64） 二维  4行*2列 4*2=8个0.
a = np.zeros((4,2));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
#0-1随机数  float
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
```

<img width="800" height="150" alt="image" src="https://github.com/user-attachments/assets/b193d455-2202-4e3a-8e68-0067131590f4" />

Some data creation routines do not take a shape tuple:
```python
np.arange([start,] stop[, step])
```
- **Parameter**：
  - `start`：Initial value (default: 0
  - `stop`：The ending value (excluding 'stop'
  - `step`：Step size (default: 1
- **return**：The array from 'start' to < stop 'increments in step size' step '.   [start,stop)

```python
#float 连续递增数字
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
#int 连续递增数字
a = np.arange(4);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# np.arange([start,] stop[, step])  start：起始值（默认 0） stop：结束值（不包含 stop） step：步长（默认 1） 返回：从 start 到 < stop 的数组，按照步长 step 递增。
#float [4,6)每次增加0.5的所有数字
a = np.arange(4,6,0.5);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
#一维 生成4个随机0-1的数字 float
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
#二维 float  2行*3列 2*3个随机0-1的数字
a = np.random.rand(2,3);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
```

<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/1f9d25c6-5c6a-413b-bbe1-c0793627fa03" />


values can be specified manually as well.
```python
np.array(object, dtype=None, ndmin=0)
```

- **Parameter**：
  - `object`：要转换为数组的列表、元组等
  - `dtype`：指定数组的数据类型（如 int, float 等）
  - `ndmin`：最小维度（比如 ndmin=2 会把一维数组变成二维）

shape returns a tuple (dim0, dim1, dim2, …)

Each dimension represents the number of elements along that axis.

- Axis 0 → length of the outermost array

- Axis 1 → length of the elements (1D arrays) in Axis 0

- Axis 2 → length of the elements (1D arrays) in Axis 1

- And so on...


```python
# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([[5,4,3,2],[5.,4,3,2]]);  print(f"np.array([[5,4,3,2],[5.,4,3,2]]):  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([[[5, 4, 3, 2],[5., 4, 3, 2]],[[1, 2, 3, 4],[1., 2, 3, 4]]]);
print(f"np.array([[[5, 4, 3, 2],[5., 4, 3, 2]],[[1, 2, 3, 4],[1., 2, 3, 4]]]: a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

```

<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/57c1bb54-3d3d-4b24-b9d6-5b0b7c0830e7" />

np.array([5,4,3,2]) and np.array([5.,4,3,2]) have all created a one-dimensional vector a with four elements. a.shape returns the dimensions. Here we see a.shape = (4,) indicating a 1-d array with 4 elements.

| 特性   | `np.array` | `np.arange`       | `np.zeros` |
| ---- | ---------- | ----------------- | ---------- |
| 数组内容 | 由已有数据决定    | 自动生成递增数列          | 全部为 0      |
| 参数   | 列表、元组等     | start, stop, step | shape      |
| 维度   | 可由数据结构决定   | 一般一维（可 reshape）   | 可指定 shape  |

## 4.4 Operations on Vectors
Let's explore some operations using vectors.
### 4.4.1 Indexing
Elements of vectors can be accessed via indexing and slicing. NumPy provides a very complete set of indexing and slicing capabilities.

**Indexing** means referring to an element of an array by its position within the array.

**Slicing** means getting a subset of elements from an array based on their indices.

NumPy starts indexing at zero so the 3rd element of an vector a is a[2].
```python
#vector indexing operations on 1-D vectors 一维向量索引操作
a = np.arange(10)
print(a)

#access an element 访问单个元素
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end  访问最后一个元素，负数索引从末尾开始计算
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce and error 索引必须在向量的范围内，否则会产生错误
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)
```

<img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/fa53d64c-1ef0-4071-8630-95ef29d31123" />

### 4.4.2 Slicing
Slicing creates an array of indices using a set of three values **(start:stop:step)**. A subset of values is also valid. 

  - `start`：Initial value (default: 0
  - `stop`：The ending value (excluding 'stop'
  - `step`：Step size (default: 1
    
```python
#vector slicing operations  向量切片操作
a = np.arange(10)
print(f"a         = {a}")
#access 5 consecutive elements (start:stop:step)  访问5个连续的元素（开始：停止：步长）
c = a[2:7:1];     print("a[2:7:1] = ", c)
# access 3 elements separated by two  访问3个元素，由两个元素分开 从2开始(包括2)，到6结束(不包括6)，步长为2
c = a[2:6:2];     print("a[2:6:2] = ", c)
# access all elements index 3 and above 访问所有索引为3及以上的元素
c = a[3:];        print("a[3:]    = ", c)
# access all elements below index 3  访问索引3以下的所有元素
c = a[:3];        print("a[:3]    = ", c)
# access all elements  访问所有元素
c = a[:];         print("a[:]     = ", c)
```

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/5cf81dec-f7cf-4268-86db-6fb084e20ed7" />


### 4.4.3 Single vector operations
There are a number of useful operations that involve operations on a single vector.
```python
a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a  a中每个元素的负数
b = -a 
print(f"b = -a        : {b}")
# sum all elements of a, returns a scalar 将a的所有元素相加，返回一个标量
b = np.sum(a)  
print(f"b = np.sum(a) : {b}")
#计算a的平均数
b = np.mean(a)
print(f"b = np.mean(a): {b}")
# a中每个元素的平方
b = a**2
print(f"b = a**2      : {b}")
```

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/6d1e713b-fc6d-4c06-8e4a-90cf163c8964" />

### 4.4.4 Vector Vector element-wise operations
Most of the NumPy arithmetic, logical and comparison operations apply to vectors as well. These operators work on an element-by-element basis. For example

$$\mathbf{a} + \mathbf{b} = \sum_{i=0}^{n-1} a_i + b_i$$
```python
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")
```

<img width="400" height="40" alt="image" src="https://github.com/user-attachments/assets/2d0ea7ee-23fe-4778-9f2e-7b50954d5816" />

```python
#try a mismatched vector operation 尝试维度不匹配的向量操作
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)
```

<img width="700" height="70" alt="image" src="https://github.com/user-attachments/assets/c646b54b-5e6e-4e85-b42a-6b4ed85d268f" />

### 4.4.5 Scalar Vector operations

Vectors can be 'scaled' by scalar values. A scalar value is just a number. The scalar multiplies all the elements of the vector.

```python
a = np.array([1, 2, 3, 4])

# multiply a by a scalar  乘以一个标量
b = 5 * a 
print(f"b = 5 * a : {b}")
```

<img width="300" height="30" alt="image" src="https://github.com/user-attachments/assets/0694bfce-d3fd-4cd2-8eb1-3e7cb8a312dd" />

### 4.4.6 Vector Vector dot product
The dot product is a mainstay of Linear Algebra and NumPy. This is an operation used extensively in this course and should be well understood. The dot product is shown below.

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/ff22404b-27f1-43f9-8631-c4ad9528cae7" />

The dot product **multiplies the values in two vectors element-wise** and then **sums the result**. 

Vector dot product requires **the dimensions of the two vectors to be the same**.

Let's implement our own version of the dot product below:

**Using a for loop**, implement a function which returns the dot product of two vectors. The function to return given inputs a and b:

$$x = \sum_{i=0}^{n-1} a_i b_i$$

Assume both a and b are the same shape.
```python
def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x
```
```python
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")
```

<img width="200" height="30" alt="image" src="https://github.com/user-attachments/assets/0628f942-b08d-48c2-b194-0a609d5bb761" />

Note, the dot product is expected to return a scalar value.

Let's try the same operations using **np.dot**.
```python
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")
```

<img width="669" height="78" alt="image" src="https://github.com/user-attachments/assets/ff060d7f-3f75-407d-8ce1-a8edc603d771" />

一维数组做dot得到的结果是纯标量（scalar）,不是数组，因此没有维度。一个真正的 0 维标量

Above, you will note that the results for 1-D matched our implementation.
### 4.4.7 The Need for Speed: vector vs for loop
We utilized the NumPy library because it improves speed memory efficiency. Let's demonstrate:

```python
np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory
```

<img width="300" height="100" alt="image" src="https://github.com/user-attachments/assets/afdead82-8e2e-4568-97a9-a0d88481f825" />


So, vectorization provides a large speed up in this example. This is because NumPy makes better use of available data parallelism in the underlying hardware. GPU's and modern CPU's implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel. This is critical in Machine Learning where the data sets are often very large.
### 4.4.8 Vector Vector operations in Course 1
Vector Vector operations will appear frequently in course 1. Here is why:

- Going forward, our examples will be stored in an array, X_train of dimension (m,n). This will be explained more in context, but here it is important to note it is a 2 Dimensional array or matrix (see next section on matrices).

- w will be a 1-dimensional vector of shape (n,).

- we will perform operations by looping through the examples, extracting each example to work on individually by indexing X. For example:X[i]

- X[i] returns a value of shape (n,), a 1-dimensional vector. Consequently, operations involving X[i] are often vector-vector.

That is a somewhat lengthy explanation, but aligning and understanding the shapes of your operands is important when performing vector operations.
```python
# show common Course 1 example
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")
```

# 5 Matrices
## 5.1 Abstract
Matrices, are two dimensional arrays. The elements of a matrix are all of the same type. In notation, matrices are denoted with capitol, bold letter such as x . In this and other labs, m is often the number of rows and n the number of columns. The elements of a matrix can be referenced with a two dimensional index. In math settings, numbers in the index typically run from 1 to n. In computer science and these labs, indexing will run from 0 to n-1.

<img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/6e0041d2-841d-46c5-954b-21e6f5fed26a" />

Generic Matrix Notation, 1st index is row, 2nd is column
## 5.2 NumPy Arrays
NumPy's basic data structure is an indexable, n-dimensional array containing elements of the same type (dtype). These were described earlier. Matrices have a two-dimensional (2-D) index [m,n].

In Course 1, 2-D matrices are used to hold training data. Training data is m examples by n features creating an (m,n) array. Course 1 does not do operations directly on matrices but typically extracts an example as a vector and operates on that. Below you will review:

- data creation
- 
- slicing and indexing
## 5.3 Matrix Creation
The same functions that created 1-D vectors will create 2-D or n-D arrays. Here are some examples

Below, the shape tuple is provided to achieve a 2-D result. Notice how NumPy uses brackets to denote each dimension. Notice further than NumPy, when printing, will print one row per line.
```python
a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}")
```

==========

One can also manually specify data. Dimensions are specified with additional brackets matching the format in the printing above.
```python
# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")
```

============

## 5.4 Operations on Matrices
Let's explore some operations using matrices.
### 5.4.1 Indexing
Matrices include a second index. The two indexes describe [row, column]. Access can either return an element or a row/column. See below:
```python
#vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
```

=========

It is worth drawing attention to the last example. Accessing a matrix by just specifying the row will return a 1-D vector.

**Reshape**

The previous example used reshape to shape the array.

a = np.arange(6).reshape(-1, 2) 

This line of code first created a 1-D Vector of six elements. It then reshaped that vector into a 2-D array using the reshape command. This could have been written:

a = np.arange(6).reshape(3, 2) 

To arrive at the same 3 row, 2 column array. The -1 argument tells the routine to compute the number of rows given the size of the array and the number of columns.

### 5.4.2 Slicing
Slicing creates an array of indices using a set of three values (start:stop:step). A subset of values is also valid. Its use is best explained by example:
```python
#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
```

=========




