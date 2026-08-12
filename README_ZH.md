# SparseView3DGS：个人图片三维重建工具

这个项目不下载或复现三篇论文的代码，而是把它们的共同思路整理成一个可替换的个人图片重建流程：

1. **多视图一致性**：先用 COLMAP 估计相机位姿和稀疏点云，拒绝无法注册的图片。
2. **深度/极线先验接口**：在 `config.yaml` 中保留深度先验与极线一致性开关，后端可以接入 Depth Anything、MiDaS 或自定义深度模型。
3. **少视图稳健初始化**：对模糊度、分辨率和注册数量生成检查报告，并为 Gaussian Splatting 后端输出标准 COLMAP 数据。

当前仓库只包含本地工具和适配器，不自带大型模型、CUDA 扩展或论文仓库。这样可以避免重复下载，也方便使用你自己的图片。

> 完整的程序流程、统一入口、断点续训、指定视角渲染和当前实验结果请参阅 [`PIPELINE_ZH.md`](PIPELINE_ZH.md)。

> 新电脑从零配置环境请参阅 [`SETUP_ZH.md`](SETUP_ZH.md)。

## 目录

```text
SparseView3DGS/
├─ data/                      # 原始照片和处理后的训练数据
├─ program/                   # FSGS、CoR-GS、COLMAP 和项目程序
├─ output/                    # COLMAP 重建、模型、检查点和渲染结果
├─ config.yaml                # 少视图/先验/匹配配置
└─ program/scripts/           # 独立训练、渲染和兼容入口
```

## 当前运行方式

### 1. 准备图片

将同一物体或场景的照片放入项目根目录的 `data/`。修改 `data/` 中的照片后，训练程序会自动重新执行 COLMAP。默认只读取 `data/` 根目录的照片，不会混入 `data/fsgs/`、`data/processed/` 等历史子目录；如确实需要递归读取，可增加 `--recursive-images`。

固定环境使用：

```text
C:\Users\leo\.conda\envs\ai\python.exe
```

### 2. 训练

训练、COLMAP 准备和模型输出由根目录的 [`train.py`](train.py) 负责。默认使用 `data/`，模型保存到项目根目录的 `model/`：

```powershell
& C:\Users\leo\.conda\envs\ai\python.exe `
  .\train.py `
  --iterations 4000
```

默认跳过需要下载 VGG16 权重的 LPIPS 评估；如本机已有权重并需要测试指标，可添加 `--eval`。

默认方法是 `corgs_fsgs`，同时启用 FSGS、CoR-GS 和 Binocular。也可以选择：

```powershell
# 仅 FSGS
& C:\Users\leo\.conda\envs\ai\python.exe .\train.py --method fsgs --iterations 4000

# CoR-GS + Binocular
& C:\Users\leo\.conda\envs\ai\python.exe .\train.py --method corgs --iterations 4000

# FSGS + CoR-GS + Binocular
& C:\Users\leo\.conda\envs\ai\python.exe .\train.py --method corgs_fsgs --iterations 4000
```

训练生成的 COLMAP/3DGS 数据位于：

```text
output/reconstruction/pipeline/dense/
```

如果已有可直接训练的 `images/` 和 `sparse/0/` 数据，可通过 `--dataset` 指定，从而跳过 COLMAP：

```powershell
& C:\Users\leo\.conda\envs\ai\python.exe `
  .\train.py `
  --dataset .\data\fsgs\latest_18_new_uniform_single_views `
  --iterations 4000
```

### 3. 固定视角渲染

渲染程序是根目录的 [`render.py`](render.py)，默认读取 `model/` 和上一次准备生成的 `output/reconstruction/pipeline/dense/`：

```powershell
& C:\Users\leo\.conda\envs\ai\python.exe `
  .\render.py `
  --iteration 4000 `
  --views view_018,view_026,view_035 `
  --render-depth
```

输出位于：

```text
model/train/ours_4000/renders/
```

### 4. 自由视角渲染

```powershell
& C:\Users\leo\.conda\envs\ai\python.exe `
  .\render.py `
  --interactive `
  --iteration 4000
```

- 鼠标左键拖动：旋转；
- 鼠标右键拖动：平移；
- 滚轮：缩放；
- `R`：重置视角；
- `S`：保存图片和相机位姿；
- `Q` 或 `Esc`：退出。

保存结果位于：

```text
model/interactive/
```

旧的 [`run_pipeline.py`](program/scripts/run_pipeline.py) 仍然保留，可用于兼容旧命令和一键流程。

## 拍摄建议

- 围绕物体缓慢移动，避免只拍正面；
- 每张图保留完整物体和稳定背景，避免强反光、透明表面和运动模糊；
- 场景静止，曝光尽量锁定；
- 少于 8 张图时优先使用有明显共同区域的角度，并降低对新视角的期待；
- 若需要尺度，可在场景中放置已知尺寸标记，COLMAP 本身只能恢复相似变换意义下的尺度。

## 与三类方法的对应关系

本工具不是论文的官方实现。`config.yaml` 的 `view_consistency`、`epipolar_depth_prior` 和 `few_shot_regularization` 是后端适配的统一配置入口，默认先完成可靠的 COLMAP 几何初始化；在拿到你的图片后，再根据场景决定是否启用具体深度模型或少视图正则化。
