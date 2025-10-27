# Cost Function

<img width="316" height="225" alt="image" src="https://github.com/user-attachments/assets/ebff62a3-087e-4d52-a2cd-4556526c2087" />

 
- **model**: $f_{w,b}(x^{(i)}) = w x^{(i)} + b$
- **$\hat{y}$**: prediction     ${\hat{y}}^{(i)} = f_{w,b}(x^{(i)}) = w x^{(i)} + b$
- **y**: target
- **m**: number of training examples
- **w,b**: parameters参数, coefficients系数, weights权重

Build a cost function that doesn't automatically get bigger as the training set size gets larger.

**Cost Function:**

$$J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}\left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2$$

To measures the difference between the model's predictions and the actual true values for y.

To find $w,b$ that ${\hat{y}}^{(i)}$ is close to $y^{(i)}$ for all $(x^{(i)}, y^{(i)})$

**How the cost function can be used to find the best parameters for model?**

- **model:** $f_{w,b}(x^{(i)}) = w x^{(i)} + b$
- **parameters:** w,b
- **cost function:** $$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2$$
- **goal:** $$\underset{w,b}{\text{minimize }} J(w, b)$$

 The goal is to find values for w and b that make J(w,b) as small as possible.

b=0

$f_{w,b}(x^{(i)}) = w x^{(i)}$

- **w=1**:

$x^{(1)} = 1, y^{(1)} = 1, f_w(x^{(1)}) = 1$

$x^{(2)} = 2, y^{(2)} = 2, f_w(x^{(2)}) = 2$

$x^{(3)} = 3, y^{(3)} = 3, f_w(x^{(3)}) = 3$

$J(1) = \frac{1}{2*3} \times [(1 - 1)^2 + (2 - 2)^2 + (3 - 3)^2] = 0$

<img width="389" height="334" alt="image" src="https://github.com/user-attachments/assets/cba5d675-5225-4e8e-8654-e4cf62fb7612" />

- **w=0.5**:

$x^{(1)} = 1, y^{(1)} = 1, f_{0.5}(x^{(1)}) = 0.5$

$x^{(2)} = 2, y^{(2)} = 2, f_{0.5}(x^{(2)}) = 1$

$x^{(3)} = 3, y^{(3)} = 3, f_{0.5}(x^{(3)}) = 1.5$

$J(0.5) = \frac{1}{2*3} \times [(0.5 - 1)^2 + (1 - 2)^2 + (1.5 - 3)^2] = \frac{7}{12}$

<img width="416" height="290" alt="image" src="https://github.com/user-attachments/assets/3ae43850-68e1-4f76-9054-f9575a62923a" />

- **w=0**:

$x^{(1)} = 1, y^{(1)} = 1, f_0(x^{(1)}) = 0$

$x^{(2)} = 2, y^{(2)} = 2, f_0(x^{(2)}) = 0$

$x^{(3)} = 3, y^{(3)} = 3, f_0(x^{(3)}) = 0$

$J(0) = \frac{1}{2*3} \times [(0 - 1)^2 + (0 - 2)^2 + (0 - 3)^2] = \frac{7}{3}$

<img width="383" height="248" alt="image" src="https://github.com/user-attachments/assets/49fe9835-da2b-4975-9e70-fd0e723ca035" />

**J(w):**

<img width="362" height="360" alt="image" src="https://github.com/user-attachments/assets/ab68257d-c694-4c24-8713-fde4a8de4728" />

w=1,

$$\underset{w=1}{\text{minimize }} J(w, b)$$


