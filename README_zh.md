# Traffic Sign AI 道路标志识别

这是一个使用 Flask 的道路标志识别研究项目。系统先用 YOLO11n 在道路场景中检测标志，再把检测区域裁剪出来，交给 326 类 CNN 分类器。

YOLO 和 CNN 在系统中负责不同工作。实验对比的是基线 CNN 和加入数据增强的 CNN；网站将 YOLO 检测器与增强版 CNN 组合使用。

> 这是研究原型，不是自动驾驶或驾驶辅助系统。请勿依据模型输出作出驾驶决定。

## 主要功能

- 上传 JPG、JPEG、PNG、WEBP 或 BMP 图片，最大 10 MB。
- 使用网站内的两张示例图片进行演示。
- 查看检测框、预测类别和 YOLO/CNN 置信度。
- 使用 CPU 推理；安装兼容 CUDA 的 PyTorch 后也可使用 GPU。

置信度不是准确率。

## 本地运行网站

安装 `requirements.txt` 中的依赖。PyTorch 与 torchvision 需要和操作系统及 CUDA 版本匹配；如需 GPU，请按 PyTorch 官方安装说明选择对应版本。

```bash
python -m pip install -r requirements.txt
python app.py
```

然后在浏览器打开 `http://127.0.0.1:5000`。Flask 开发服务器只用于本地演示。

网站需要在 `models/` 中找到以下文件：

- `yolo_best.pt`
- `cnn_augmentation_best.pth`
- `final_class_to_index.json`

模型权重已获准纳入项目仓库，并会随代码一同发布。完整 MTSD 图片和标注不放入仓库；请按下方数据说明从官方来源获取。

## 训练与复现

- [`notebooks/MTSD_Object_Detection.ipynb`](notebooks/MTSD_Object_Detection.ipynb)：MTSD 标注检查、YOLO 数据准备、训练和验证。
- [`notebooks/02_CNN_Training_and_Experiments.ipynb`](notebooks/02_CNN_Training_and_Experiments.ipynb)：从本机 IPython 执行历史中找回的 CNN 数据处理、基线训练、增强训练与结果绘图代码。
- [`scripts/`](scripts/)：原项目的独立推理脚本。
- [`results/`](results/)：训练历史、验证结果、图表、配置和数据质量记录。
- [`data/manifests/`](data/manifests/)：类别映射、数据清单和处理报告。

完整 MTSD 图片不包含在仓库中。请从[官方 Mapillary MTSD 页面](https://www.mapillary.com/dataset/trafficsign)获取，并遵守数据附带的许可和署名要求。先阅读 [`data/README.md`](data/README.md)，再按笔记本说明设置本机路径。独立脚本可通过 `MTSD_PROJECT_ROOT` 环境变量指定本机数据目录。

恢复出的 CNN 训练设置包括：326 类、64 × 64 输入、batch size 64、20 个 epoch、随机种子 42、Adam 学习率 0.001 和交叉熵损失。增强流程包括最多 10 度旋转、最多 5% 平移、0.95–1.05 缩放，以及亮度/对比度/饱和度扰动。更多来源说明见 [`docs/SOURCE_RECOVERY.md`](docs/SOURCE_RECOVERY.md)。

## 项目记录中的验证结果

| 实验 | 验证结果 |
|---|---:|
| 基线 CNN | 最佳验证准确率 90.66%，第 19 轮 |
| 数据增强 CNN | 最佳验证准确率 92.91%，第 17 轮 |
| YOLO11n | precision 0.73038，recall 0.56097，mAP@50 0.60818，mAP@50–95 0.40564 |

CNN 项目记录包含 60,162 张训练裁剪图、8,736 张验证裁剪图和 326 个筛选类别。所有上述分数都是验证集结果，不是独立测试集结果。详情和原始结果文件见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

本项目的 326 类是根据 MTSD 标签筛选得到的自定义映射，不等同于论文中的官方 313 类 taxonomy。

## 数据、测试集与发布说明

本机有 10,544 张 MTSD test 图片，但官方不公开其标注以保证评估公平。因此本项目不报告独立测试指标，只报告验证结果。参见 [MTSD 论文](https://arxiv.org/abs/1909.04422)。

MTSD 来源及署名说明见 [`data/README.md`](data/README.md)。本项目代码依照 AGPL-3.0 发布，模型权重与演示图片也随仓库提供。许可和第三方组件说明见 [`LICENSE`](LICENSE) 与 [`docs/LICENSING.md`](docs/LICENSING.md)。

## 许可证

本项目依照 AGPL-3.0 发布，详见 [`LICENSE`](LICENSE)。数据来源与署名说明见 [`docs/LICENSING.md`](docs/LICENSING.md)。
