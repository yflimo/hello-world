git clone git@github.com:yflimo/hello-world.git
→ 从 GitHub 上把远程仓库复制到本地。

cd hello-world
→ 进入项目目录。

git checkout -b ff_test
→ 基于当前分支新建并切换到 ff_test 分支，用来写你的新功能。

uv venv
→ 在项目里创建一个虚拟环境，保证依赖和系统隔离。

source .venv/bin/activate
→ 激活虚拟环境，后续的 Python/包安装都只作用在这个环境里。

mkdir ff_unit1 + cd ff_unit1
→ 新建一个功能目录 ff_unit1，进入其中。

touch learn_tensor.py
→ 新建一个空的 Python 文件。

vim learn_tensor.py
→ 打开编辑器，写代码。

python learn_tensor.py
→ 运行测试代码。

git add learn_tensor.py
→ 把新文件放入 Git 暂存区。

git commit -m "to learn tensor"
→ 提交更改，并写提交说明。

git push --set-upstream origin ff_test
→ 把你本地的 ff_test 分支推送到远程，并建立本地分支和远程分支的关联。
