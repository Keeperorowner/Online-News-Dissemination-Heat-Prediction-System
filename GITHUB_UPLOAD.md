# 如何将项目上传到指定的GitHub仓库

## 前提条件

1. 确保已在本地安装Git
2. 确保已拥有GitHub账户（Keeperorowner）
3. 确保项目文件已准备就绪

## 方法一：使用部署脚本（推荐）

项目目录中已包含一个部署脚本 [deploy_to_github.bat](file:///d%3A/.Temp/test/deploy_to_github.bat)，您可以直接双击运行该脚本，它会自动执行以下操作：

1. 初始化Git仓库
2. 添加所有文件到暂存区
3. 提交文件
4. 添加远程仓库地址：https://github.com/Keeperorowner/Online-News-Dissemination-Heat-Prediction-System.git
5. 推送到GitHub

## 方法二：手动执行命令

如果您想手动执行命令，请打开命令行工具（如CMD、PowerShell或Git Bash），然后执行以下命令：

```bash
# 进入项目目录
cd /path/to/your/project

# 初始化Git仓库
git init

# 添加所有文件到暂存区
git add .

# 提交文件
git commit -m "Initial commit: Online News Dissemination Heat Prediction System"

# 添加远程仓库地址
git remote add origin https://github.com/Keeperorowner/Online-News-Dissemination-Heat-Prediction-System.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

## 故障排除

如果在推送过程中遇到问题，请尝试以下解决方案：

1. 如果提示仓库已存在远程内容，可以先拉取远程内容：
   ```bash
   git pull origin main --allow-unrelated-histories
   ```

2. 如果用户名或密码验证失败，请考虑使用GitHub Personal Access Token：
   - 在GitHub上生成一个新的Personal Access Token
   - 使用Token代替密码进行身份验证

3. 如果遇到网络问题，请检查网络连接或尝试使用代理

## 项目使用说明

其他人可以通过以下方式使用您的项目：

1. 克隆仓库：
   ```bash
   git clone https://github.com/Keeperorowner/Online-News-Dissemination-Heat-Prediction-System.git
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 下载数据集并按README.md中的说明放置到正确位置

4. 运行模型：
   ```bash
   python news_popularity_classifier.py
   ```