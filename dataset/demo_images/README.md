# 现场演示用图（未参与训练）

以下 6 张图片全部来自 test 划分，训练与验证阶段均未使用，符合文档一现场验收要求（至少备 3 张未参与训练的真实图）。

| 文件名 | 类别 | manifest 划分 | sha256 |
|---|---|---|---|
| cardboard_00394.jpg | cardboard | test | `2185821a80c36b8a...` |
| glass_00495.jpg | glass | test | `d4b92d02a37b07af...` |
| metal_00402.jpg | metal | test | `40fa0ff0134ff13c...` |
| paper_00549.jpg | paper | test | `345de671ff883879...` |
| plastic_00470.jpg | plastic | test | `ced8ff08c7660be7...` |
| trash_00130.jpg | trash | test | `03db7bbb8f341665...` |

验证方式：`dataset/dataset_manifest.csv` 中每条的 `split` 字段均为 `test`，可用 image_id 检索核对。
