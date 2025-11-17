# Multiple Features

- $x_j$ = $j^{th}$ feature  
- $n$ = number of features  
- $\vec{x}^{(i)}$ = features of the $i$-th training example  
- $x_j^{(i)}$ = value of feature $j$ in the $i$-th training example

$\vec{x}^{(i)}$ is not a number,but it's actually a list of number that is a **vector**.

### multiple linear regression

$f_{\vec{w},\, b}(\vec{x})
= w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$

- $\vec{w} = [w_1,w_2,w_3, \dots\,w_n]$

- b is a number

- $\vec{x} = [x_1,x_2,x_3,\dots\,x_n]$

$f_{\vec{w},\, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$


# Vectorization

parameters and features

- $\vec{w} = [w_1,w_2,w_3]$

- b is a number

- $\vec{x} = [x_1,x_2,x_3]$

linear algebra:count from **1**

```python
w=np.array([1.0,2.5,-3.3])
b=3
x=np.array([10,20,30])
```

code count from **0**

#### Without vectorization🙁

$f_{\vec{w},\, b}(\vec{x})= w_1 x_1 + w_2 x_2 + w_3 x_3 + b$

```python
f=w[0]*x[0]+
  w[1]*x[1]+
  w[2]*x[2]+b
```

#### Without vectorization😐

$f_{\vec{w}, b}(\vec{x}) = \sum_{\substack{j=1 \\ ~}}^{n} w_j x_j + b$

```python
f=0
for j in range(0,n):
   f=f+w[j]*x[j]
f=f+b
```

#### Vectorization😀 

$f_{\vec{w},\, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$

```python
f=np.dot(w,x)+b
```

makes code shorter;easier to write;easier to read;makes it run much faster


- 0)get all values of thee vectors w and x,

- 1)multiplies each pair of w and x with each other all at the same time in parallel

- 2)the computer takes these numbers and use specialized hardware to add together very efficiently








