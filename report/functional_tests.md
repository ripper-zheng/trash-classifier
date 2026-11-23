| 编号 | 用例 | 预期 | 实测结果 |
|---|---|---|---|
| T01 | 上传正常 JPG 图片 | 返回类别与置信度 | HTTP 200 · class=cardboard, conf=0.4465 |
| T02 | 上传 PNG 图片 | 正常推理 | HTTP 200 · class=metal, conf=0.3785 |
| T03 | 上传 .txt 文件 | 拒绝并提示 | HTTP 400 · 不支持的文件类型，仅允许 JPG/PNG/WEBP |
| T04 | 上传超过 10 MB 文件 | 返回 413 提示 | HTTP 413 · Flask 返回 413 Request Entity Too Large 错误页 |
| T05 | 改扩展名的文本文件 | 格式校验拦截 | HTTP 400 · 图片损坏或格式不合法 |
| T06 | 未携带 file 字段提交 | 提示未收到文件 | HTTP 400 · 未收到文件字段 file |
| T08 | 上传未参与训练的真实图片 | 返回合理类别 | HTTP 200 · class=glass, conf=0.3921 |
| T07 | 模型权重缺失时启动服务 | 服务拒绝启动 | 代码审查确认：app.py 加载失败即 sys.exit(1)，见 5.4 节 |
