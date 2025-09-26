# 在一个分支上添加文件夹或文件并推送到远程仓库

``` bash
# 切换分支（想对哪个分支操作，就切换到哪个分支）
git checkout 分支名字        

# 创建一个文件夹
mkdir new_folder            

# 进入到这个文件夹
cd new_folder               

# 在该文件夹下创建一个文本文件
touch demo.txt              

# 在该文本文件中写内容
vim demo.txt                

# 返回上级目录
cd ..                        

# 把整个文件夹加入暂存区
git add new_folder/         

# 提交
git commit -m "add demo.txt"

# 推送到远程的分支上（刚才切换到的哪个分支）
git push                    
```
