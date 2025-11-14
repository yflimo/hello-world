# Gradient Descent

**1) Start with some w,b(set w=0,b=0) as the initial guess**

**2) Keep changing w,b to reduce J(w,b)**

**3) Until settle at or near a minimum (may have >1 minimum)**


**Apply：**
- Linear Regression  （For linear regression with the squared error cost function always end up with a bowl shape or a hammock shape.

- Training for some of the most advanced neural network models.(Deep learning models)


the direction of steepest descent    ->...(repeat the process)->    until find local minimum

choose a different starting point will end up in different minimum.

start going down the first minimum ,gradient descent won't lead to the second minimum.And the same is ture if started going down the second minimum.

## Gradient Descent Algorithm

**tmp_w = w - α $\frac{d}{dw}$ J(w,b)    (1)**

**tmp_b = b - α $\frac{d}{db}$ J(w,b)    (2)**

**w = tmp_w                    (3)**

**b = tmp_b                    (4)**

**α：** learning rate;0-1;usually a small positive number between 0 and 1;control how big of a step take downhill;

(3) and (4) must come after (1) and (2) to ensure **simultaneous updates**

Repeat (1)-(4) until **convergence**

**convergence:** reach the point at a local minimum where the parameters w and b no longer change much with each additional step that take.

## Gradient Descent Intuition

<img width="350" height="400" alt="image" src="https://github.com/user-attachments/assets/4fda59a9-56fd-4dc4-8c7b-7ed55bb357f9" />

**b=0:  w = w - α $\frac{d}{dw}$ J(w)**


**1) $\frac{d}{dw}$ J(w)>0 ,k>0 ,0<α<1 ,w↓ ,w moves to the right**

     w=w-α*(positive number)
   
**2) $\frac{d}{dw}$ J(w) ,k<0 ,0<α<1 ,w↑ ,w moves to the left**

     w=w-α*(negative number)

In both 1) and 2), w is moving in the direction where J(w) is decreasing

## Learning Rate α

Can reach local minimum with **fixed** learning rate.

w = w - α $\frac{d}{dw}$ J(w)

k↓, $\frac{d}{dw}$ J(w) ↓,Δw↓,until $\frac{d}{dw}$ J(w)=0

- If α too **small**, Gradient descent may be slow

- If α too **large**, Gradient descent may Overshoot,never reach minimum; Fail to converge,even diverge;

As we get nearer a local minimum, gradient descent will automatically take smaller steps, because as approach the local minimum,the derivative automatically gets smaller.

 **Near a local minimum:**
 
 **-Derivative becomes smaller,** 
 
 **-Update steps becomes smaller.**


 ## Gradient Descent for Linear Regression

- **Linear regression model:** $f_{w,b}(x^{(i)}) = w x^{(i)}+b$
- **Cost function:** $$J(w) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w}(x^{(i)}) - y^{(i)}\right)^2$$
- **Gradient Descent algorithm :**
  
    repeat until convergence{

    w = w - α $\frac{d}{dw}$ J(w,b)

    b = b - α $\frac{d}{db}$ J(w,b)

    }

$\frac{d}{dw} J(w,b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}$

$\frac{d}{db} J(w,b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)$


repeat until convergence{

$w = w - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}$

$b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)$

}

<img width="350" height="400" alt="image" src="https://github.com/user-attachments/assets/0eeb06b7-7527-47bb-925c-df513df44510" />

<img width="350" height="400" alt="image" src="https://github.com/user-attachments/assets/40b7d6ac-70d8-4f96-82bb-6ecd008d75f6" />

<img width="350" height="400" alt="image" src="https://github.com/user-attachments/assets/627870d3-4139-40fe-be81-4dacff7a788c" />
