# 垃圾分类六分类系统（深度学习案例课程项目）

基于 PyTorch 的垃圾分类图像识别系统，使用 TrashNet 六分类数据，完成真实数据整理、ResNet18 迁移学习、TensorBoard 训练监控、Flask 前端实时推理与（可选）Docker 部署的全链路。

> ⚠️ 提交前须知：本仓库所有命令、指标、截图都必须由你本人真实运行产出。项目设计报告中的示例值（如 3211 张、91.5% 等）不得直接复制，须替换为你的真实运行结果。

## 1. 项目信息

- 课程：深度学习案例
- 选题：垃圾分类六分类（cardboard / glass / metal / paper / plastic / trash）
- 源码仓库：见 `source_repository.txt`（提交前填写真实地址）

## 2. 环境安装

```bash
# 本机实测 Python 3.13.14；torch 2.6.0 + torchvision 0.21.0 兼容 3.13
# Windows:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Linux/Mac: source .venv/bin/activate
```

安装后记录实测版本（写入报告第 11 章）：

```bash
pip freeze > installed_versions.txt
```

查看并记录实际版本（写入报告）：

```bash
python --version
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available())"
```

## 3. 数据获取

数据来自公开数据集 TrashNet：<https://github.com/garythung/trashnet>

实测可用文件为 `data/dataset-resized.zip`（约 41 MB，2527 张六分类图像）。
脚本通过 GitHub API 原始内容端点下载（网络代理环境下 raw.githubusercontent 常被拦截）：

```bash
python src/download_data.py --output dataset/raw
```

下载后按要求清洗、分层划分（详见 `dataset_manifest.csv`）：

```bash
python src/prepare_data.py --raw dataset/raw --out dataset --seed 42
```

该脚本会：感知哈希去重 → 分层划分 train/val/test（70/15/15）→ 生成 `dataset/dataset_manifest.csv` → 从 val/test 抽取 10 张样例图到 `dataset/samples/`（每类至少 1 张并标注类别）。

**本机实测结果（写入报告第 3 章）：**

| 类别 | 原始 | 去重后 | train | val | test |
|---|---|---|---|---|---|
| cardboard | 403 | 395 | 276 | 59 | 60 |
| glass | 501 | 497 | 347 | 74 | 76 |
| metal | 410 | 403 | 282 | 60 | 61 |
| paper | 594 | 550 | 385 | 82 | 83 |
| plastic | 482 | 471 | 329 | 70 | 72 |
| trash | 137 | 135 | 94 | 20 | 21 |
| **合计** | **2527** | **2451** | **1713** | **365** | **373** |

共移除重复图片 76 张（感知哈希汉明距离 ≤5），无损坏/过小图片。

> 数据体量较大，不随源码入库。提交压缩包时只需带 `dataset_manifest.csv` 与 `dataset/samples/`（10 张），完整图像可用本脚本重新获取，保证可复现。

## 4. 模型训练

三组对照实验（统一 seed=42、224×224、Batch=32）：

| 实验 | 命令 | 说明 |
|---|---|---|
| E1 baseline | `python src/train.py --exp E1_cnn` | 自定义 CNN，从零训练，建立基线 |
| E2 冻结主干 | `python src/train.py --exp E2_resnet18_frozen` | ResNet18 冻结卷积层，只训分类头 |
| E3 全量微调 | `python src/train.py --exp E3_resnet18_finetune` | ResNet18 全量微调，产出 `models/best_model.pth` |

每次实验独立写入 `runs/<实验名>/`，互不覆盖。最佳权重由验证集 Macro-F1 选出，保存到 `models/best_model.pth`。

## 5. 查看训练曲线

```bash
tensorboard --logdir runs --port 6006
# 浏览器打开 http://localhost:6006
```

报告中须插入 Loss 与准确率曲线截图，并在演示视频中现场展示同一曲线。

## 6. 模型评估

```bash
python src/evaluate.py --checkpoint models/best_model.pth --out report
```

输出测试集 Accuracy / Precision / Recall / Macro-F1、6×6 混淆矩阵图、Top-3 结果以及正确/错误案例，供报告第 7、9 章使用。

## 7. 单张图片推理

```bash
python src/predict.py --image path/to/test.jpg --checkpoint models/best_model.pth
```

## 8. 前端启动

```bash
python src/app.py
# 浏览器打开 http://127.0.0.1:5000
```

前端页面实时调用已保存的 `best_model.pth` 推理，展示类别、置信度、Top-3 和推理耗时，结果不在 JavaScript 中写死。模型加载失败时服务拒绝启动。

接口：

- `GET /` 首页
- `POST /predict` 上传图片并推理，返回 JSON
- `GET /health` 服务与模型状态
- `GET /model-info` 模型名称、类别与输入尺寸

## 9. （可选）Docker 部署

```bash
docker build -t trash-classifier:v1 .
docker run -d --name trash-app -p 5000:5000 trash-classifier:v1
docker logs trash-app
curl http://localhost:5000/health
```

Docker 为加分项，非硬性验收要求。

## 10. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 准确率异常接近 100% | 训练与测试存在数据泄漏 | 检查 manifest 划分、按哈希去重后重新划分 |
| 模型加载 size mismatch | 类别数或结构不一致 | 统一使用 `src/config.py` 的类别映射与 `src/model.py` |
| 前端预测失败 | 预处理与训练不一致 | 确认训练和推理都从 `src/config.py` 读取 transforms |
| TensorBoard 无曲线 | 日志路径错误 | 确认 `runs/<实验名>/` 存在事件文件 |

## 11. 已知问题

- 类别固定为六类，无法识别电池、药品、厨余等其他类别。
- 对低照度、遮挡、透明物体与混合垃圾场景识别不稳定，系统对低置信度结果提示人工复核。
