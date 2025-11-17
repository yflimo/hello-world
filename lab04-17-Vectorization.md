# Vectorization

```python
import numpy as np    # it is an unofficial standard to use np for numpy
import time
```

## Goals
- Review the features of NumPy and Python

## Useful References
- NumPy Documentation including a basic introduction:NumPy.org
- A challenging feature topic:NumPy Broadcasting

## Python and NumPy
Python is the programming language we will be using in this course. It has a set of numeric data types and arithmetic operations. NumPy is a library that extends the base capabilities of python to add a richer data set including more numeric types, vectors, matrices, and many matrix functions. NumPy and python work together fairly seamlessly. Python arithmetic operators work on NumPy data types and many NumPy functions will accept python data types.

## Vectors

### Abstract
<img width="634" height="276" alt="image" src="https://github.com/user-attachments/assets/9b650d75-9a8e-477d-a287-01b745657307" />

Vectors, as you will use them in this course, are ordered arrays of numbers. In notation, vectors are denoted with lower case bold letters such as x. The elements of a vector are all the same type. A vector does not, for example, contain both characters and numbers. The number of elements in the array is often referred to as the dimension though mathematicians may prefer rank. The vector shown has a dimension of n. The elements of a vector can be referenced with an index. In math settings, indexes typically run from 1 to n. In computer science and these labs, indexing will typically run from 0 to n-1. In notation, elements of a vector, when referenced individually will indicate the index in a subscript, for example, the $0^{th}$ element, of the vector x is x_0. Note, the x is not bold in this case.
