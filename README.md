# Git 学习与项目操作记录

## 初始化项目

```bash
git clone git@github.com:yflimo/hello-world.git   # 从 GitHub 上把远程仓库复制到本地
cd hello-world                                    # 进入项目目录
```

## 分支管理

```bash
git checkout -b ff_test   # 基于当前分支新建并切换到 ff_test 分支，用来写新功能
```

## 创建虚拟环境

```bash
uv venv                     # 在项目里创建一个虚拟环境，保证依赖和系统隔离
source .venv/bin/activate   # 激活虚拟环境，后续操作只作用在这个环境里
```

## 功能目录与代码

```bash
mkdir ff_unit1          # 新建一个功能目录
cd ff_unit1             # 进入目录

touch learn_tensor.py   # 新建一个空的 Python 文件
vim learn_tensor.py     # 打开编辑器写代码

python learn_tensor.py  # 运行测试代码
```

## Git 操作

```bash
git add learn_tensor.py                  # 把新文件加入 Git 暂存区
git commit -m "to learn tensor"          # 提交更改，并写提交说明
git push --set-upstream origin ff_test   # 推送到远程，并建立本地和远程分支的关联
```



# Machine Learning 机器学习

Field of study that gives computers the ability to learn without being explicitly programmed.

Divided into **Supervised Learning** and **Unsupervised Learning**.

---

## Supervised Learning 监督学习

- X -> Y  or  input -> output  
- X-to-Y or input-output mappings  
- Learns from being given “right answers”.  
  - “Right answers” means given the correct label Y for input X.  

Supervised learning is divided into **Regression** and **Classification**.

### Regression 回归
- Predict a number from infinitely many possible numbers.  

**Linear Regression 线性回归**  
- Fitting a straight line.  

**Nonlinear Regression 非线性回归**  
- Fitting a curve.  

### Classification 分类
- Predict categories/classes from a small, finite set of possible outputs.  
- Fit a decision boundary.  

---

## Unsupervised Learning 无监督学习

- Data only comes with input X, but not output labels Y.  
- Algorithm has to find structure in the data.  
- Not giving the algorithm the right answer for the examples in advance.  

### Clustering 聚类
- Group similar data points together.  
- Takes data without labels and tries to automatically group them into clusters.  

### Anomaly Detection 异常检测
- Find unusual data points.  

### Dimensionality Reduction 降维
- Compress data using fewer numbers.  
