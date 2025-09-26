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
