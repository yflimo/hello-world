## venv（virtual environment)：给每个 Python 项目一个独立的、隔离的运行空间.

### **uv venv** 

就是在项目里创建这样一个隔离环境。

生成一个名为 .venv 的文件夹，里面装的就是一个独立的 Python 解释器和库路径。


### **.venv\Scripts\activate     (Windows(PowerShell))**

source .venv/bin/activate  (Linux or macOS)

进入这个环境。

### **uv pip install -e.**

用 -e . 以“可编辑模式”安装当前项目（这样修改源码后不必重新安装）。

### **deactivate** 

退出。命令行前面的 (.venv) 就会消失，返回系统环境。

### **rm -rf .venv/**  

是删除整个虚拟环境文件夹


<img width="500" height="389" alt="image" src="https://github.com/user-attachments/assets/1cbeb0c0-e8bf-48d0-86c9-9585a78fc438" />


dataset/Boston.csv                	存放数据文件

README.md         	                项目说明文件，介绍项目背景、使用方法等

boston_house_price_prediction.py		实现模型训练与预测的主要逻辑

main.py		                          程序入口文件，用来运行整个项目

plotting.py	                      	负责绘图

pyproject.toml	                    配置文件	Python项目配置文件，通常定义依赖、项目元数据
