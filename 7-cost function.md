# Cost Function

<img width="316" height="225" alt="image" src="https://github.com/user-attachments/assets/ebff62a3-087e-4d52-a2cd-4556526c2087" />

 
- **model**: $f_{w,b}(x^{(i)}) = w x^{(i)} + b$
- **y-hat**: prediction     ${\hat{y}}^{(i)} = f_{w,b}(x^{(i)}) = w x^{(i)} + b$
- **y**: target
- **m**: number of training examples
- **w,b**: parameters参数, coefficients系数, weights权重

find $w,b$ :  ${\hat{y}}^{(i)}$ is close to $y^{(i)}$ for all $(x^{(i)}, y^{(i)})$

Build a cost function that doesn't automatically get bigger as the training set size gets larger.

$$J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}\left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2$$
