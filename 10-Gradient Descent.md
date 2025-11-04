# Gradient Descent

**1) Start with some w,b(set w=0,b=0) as the initial guess**

**2) Keep changing w,b to reduce J(w,b)**

**3) Until settle at or near a minimum (may have >1 minimum)**


**Apply：**
- linear regression  （For linear regression with the squared error cost function always end up with a bowl shape or a hammock shape.

- training for some of the most advanced neural network models.(Deep learning models)


the direction of steepest descent    ->...(repeat the process)->    until find local minimum

choose a different starting point will end up in different minimum.

start going down the first minimum ,gradient descent won't lead to the second minimum.And the same is ture if started going down the second minimum.

# Gradient Descent Algorithm

```markdown
tmp_w = w - α∂/∂w J(w, b)    (1)
tmp_b = b - α∂/∂b J(w, b)    (2)
w = tmp_w                    (3)
b = tmp_b                    (4)
```

α：learning rate;0-1;usually a small positive number between 0 and 1;control how big of a step take downhill;

(3) and (4) must come after (1) and (2) to ensure **simultaneous updates**

Repeat the above operation until **convergence**

**convergence:** reach the point at a local minimum where the parameters w and b no longer change much with each additional step that take.
