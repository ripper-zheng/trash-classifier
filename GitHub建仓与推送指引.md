# GitHub 仓库创建与推送指引（最新状态）

> 本地仓库已就绪：**16 次提交、工作区 clean、66 个文件、所有交付文件已跟踪**。
> 远程建仓接口权限不足（403），故需你本人做一次网页操作；推送随后一条命令完成。

## 一、网页创建仓库（约 1 分钟）

1. 打开 <https://github.com/new>，确认登录账号是 `ripper-zheng`。
2. Repository name 填：`trash-classifier`
3. Description 可填：基于 PyTorch ResNet18 的垃圾分类六分类系统
4. 选 **Public**（文档一要求源码链接可直接访问）。
5. **不要**勾选 “Add a README file” / “.gitignore” / “license”（本地已有，避免冲突）。
6. 点 **Create repository**。

## 二、推送（一条命令搞定）

建仓后，在项目目录执行：

```powershell
cd C:\Users\Administrator\Desktop\work\trash-classifier
git remote add origin https://github.com/ripper-zheng/trash-classifier.git
git branch -M main
git push -u origin main
```

若 `git push` 弹出登录窗口，用浏览器登录 GitHub 授权；若用 Personal Access Token，在密码处粘贴 token（需 repo 权限）。

推送成功验证：

```powershell
git log --oneline -5
```

应能看到 `d594c07`、`47cd3b5` 等 16 次提交（**完整提交历史**，满足硬性要求）。

## 三、源码链接（已预填，无需再改）

`source_repository.txt` 已填好真实地址 `https://github.com/ripper-zheng/trash-classifier`，推送后即可访问。

## 四、之后保持完整提交历史

训练完成后自动生成的报告终稿、docx、混淆矩阵等新文件，提交并推送即可：

```powershell
git add -A && git commit -m "Add final report with real experiment results"
git push
```

## 五、验收现场检查

- 仓库 Public，浏览器可直接打开；
- `git log` 有多次提交，非单一最终版本；
- README 中的运行命令与仓库代码一致；
- `source_repository.txt` 地址可访问。

## 附：为什么这一步无法代办

建仓接口返回 403（token 无建仓权限），本机未安装 gh 命令行、也未存 GitHub 凭证，因此网页建仓必须本人操作。本地代码、提交历史、推送命令均已准备好。
