# GitHub 仓库创建与推送指引

本地 Git 仓库已完成首次提交（commit `33dc699`，24 个文件）。
因当前 GitHub 接口权限不足（403），远程建仓需你本人操作，步骤如下。

## 一、网页创建仓库（约 1 分钟）

1. 打开 <https://github.com/new>，确认登录账号是 `ripper-zheng`。
2. Repository name 填：`trash-classifier`
3. Description 可填：基于 PyTorch ResNet18 的垃圾分类六分类系统
4. 选择 **Public**（文档一要求源码链接可直接访问；私有库须提前给评阅教师授权）。
5. **不要**勾选 “Add a README file”、”.gitignore”、”license”（本地已有，避免冲突）。
6. 点击 **Create repository**。

## 二、推送本地代码

创建后 GitHub 会显示仓库地址。在项目目录执行（PowerShell 或 Git Bash）：

```powershell
cd C:\Users\Administrator\Desktop\work\trash-classifier
git remote add origin https://github.com/ripper-zheng/trash-classifier.git
git branch -M main
git push -u origin main
```

若 `git push` 弹出登录窗口，用浏览器登录 GitHub 完成授权；若使用 Personal Access Token，在密码处粘贴 token（需 repo 权限）。

推送成功后验证：
```powershell
git log --oneline -3
```
应能看到 `33dc699 Initial commit` 及后续提交。

## 三、填写源码链接

推送成功后，编辑 `source_repository.txt`：

```
平台：GitHub
项目地址：https://github.com/ripper-zheng/trash-classifier
仓库可见性：公开
```

然后提交并推送：
```powershell
git add source_repository.txt
git commit -m "Update repository URL"
git push
```

## 四、保持完整提交历史（硬性要求）

课程要求"完整提交历史，不接受仅上传一个 ZIP 或只有最终版本"。
之后每完成一个阶段就提交一次，例如：

```powershell
git add . && git commit -m "Add data preprocessing and manifest"
git add . && git commit -m "Add training experiments E1-E3"
git add . && git commit -m "Add frontend and Flask API"
git add . && git commit -m "Add report and figures"
git push
```

这样评阅时能看到从骨架到成品的演进过程。

## 五、验收现场检查

- 仓库 Public，浏览器可直接打开；
- `git log` 有多次提交，非单一最终版本；
- README 中的运行命令与仓库代码一致。
