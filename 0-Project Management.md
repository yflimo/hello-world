## venv（virtual environment)：给每个 Python 项目一个独立的、隔离的运行空间.

### **uv venv** 

就是在项目里创建这样一个隔离环境。

生成一个名为 .venv 的文件夹，里面装的就是一个独立的 Python 解释器和库路径。


### **source .venv/bin/activate  (Linux or macOS)**

### **.venv\Scripts\activate     (Windows(PowerShell))**

进入这个环境。


### **uv pip install -e.**

用 -e . 以“可编辑模式”安装当前项目（这样修改源码后不必重新安装）。

### **deactivate** 

退出。命令行前面的 (.venv) 就会消失，返回系统环境。

### **rm -rf .venv/**  

是删除整个虚拟环境文件夹
