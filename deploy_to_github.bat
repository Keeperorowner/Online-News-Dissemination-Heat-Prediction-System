@echo off
echo 正在部署项目到GitHub仓库...

echo 步骤1: 初始化Git仓库
git init

echo 步骤2: 添加所有文件到暂存区
git add .

echo 步骤3: 提交文件
git commit -m "Initial commit: Online News Dissemination Heat Prediction System"

echo 步骤4: 添加远程仓库地址
git remote add origin https://github.com/Keeperorowner/Online-News-Dissemination-Heat-Prediction-System.git

echo 步骤5: 推送到GitHub
git branch -M main
git push -u origin main

echo 部署完成！
pause