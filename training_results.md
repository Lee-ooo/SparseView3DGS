# SparseView3DGS 训练结果

本文记录当前数据集上的 FSGS 和 CoR-GS 重建结果。

## 1. 数据与环境

- 输入照片：`C:\mine\git\SparseView3DGS\data` 中共 8 张。
- COLMAP 注册结果：8 张照片中有 5 张成功注册：`view_001`、`view_002`、`view_004`、`view_005`、`view_008`。
- 当前实际参与 SfM/训练的有效视角为 5 个，未注册的 3 张照片没有直接用于相机监督。
- Conda 环境：`ai`
- Python：`C:\Users\leo\.conda\envs\ai\python.exe`
- PyTorch：`2.11.0+cu130`
- CUDA：`13.0`
- GPU：NVIDIA GeForce RTX 4060

两个方法均复用了已编译的 CUDA 扩展，没有新建 Conda 环境，也没有清理已下载的软件包。

## 2. FSGS 结果

### 训练配置

- 方法：FSGS（Gaussian Unpooling、伪视角机制和深度正则化框架）。
- 训练迭代：2000。
- 初始点云：COLMAP 结果生成的约 80 个点。
- 输出目录：[`output/fsgs/latest_8_fsgs_2000`](C:/mine/git/SparseView3DGS/output/fsgs/latest_8_fsgs_2000)
- 最终点云：[`point_cloud.ply`](C:/mine/git/SparseView3DGS/output/fsgs/latest_8_fsgs_2000/point_cloud/iteration_2000/point_cloud.ply)
- 最终 Gaussian 数量：约 15,846 个。
- 渲染结果：[`FSGS view_001`](C:/mine/git/SparseView3DGS/output/fsgs/latest_8_fsgs_2000/train/ours_2000/renders/view_001.png)

### 重要限制

由于当前环境无法下载 MiDaS 深度模型，实际运行时使用了：

```text
--depth_weight 0
--depth_pseudo_weight 0
```

因此本次 FSGS 结果主要体现 Gaussian Unpooling 和 RGB 重建效果，完整的单目深度正则化没有启用。

## 3. CoR-GS 结果

### 训练配置

- 方法：CoR-GS（双 Gaussian 场、Pseudo-view Co-Regularization 和 Co-Pruning）。
- 两个 Gaussian 场：`GS0` 和 `GS1`。
- 训练迭代：3000。
- 第 2000 次迭代后启用伪视角协同正则化。
- 为适配 RTX 4060，伪视角协同正则改为每 5 次训练迭代调用一次：

```text
--gaussiansN 2
--coreg
--coprune
--sample_pseudo_interval 5
```

- Co-Pruning 默认每 500 次执行一次，匹配阈值使用默认值 5。
- 输出目录：[`output/corgs/latest_8_cor_gs_3000`](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_3000)
- GS0 点云：[`GS0 point_cloud.ply`](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_3000/point_cloud/iteration_3000/point_cloud.ply)
- GS1 点云：[`GS1 point_cloud.ply`](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_3000/point_cloud_gs2/iteration_3000/point_cloud.ply)
- GS0 Gaussian 数量：约 133,820 个。
- GS1 Gaussian 数量：约 134,604 个。
- GS0 渲染结果：[`CoR-GS view_001`](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_3000/train/ours_3000/renders/view_001.png)
- 深度结果：[`CoR-GS view_001 depth`](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_3000/train/ours_3000/renders/view_001_depth.png)
- 检查点：[`chkpnt3000.pth`](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_3000/chkpnt3000.pth)

### CoR-GS 适配说明

官方 CoR-GS 代码主要针对标准数据集目录结构。本次额外适配了：

- 通用 COLMAP 数据集路径读取；
- 缺少 `poses_bounds.npy` 时的默认 bounds；
- COLMAP PLY 缺少法线字段时的兼容处理；
- 复用现有 CUDA 扩展；
- 删除未实际使用的 `kmeans1d` 导入。

官方论文与代码：

- [CoR-GS 论文](https://arxiv.org/abs/2405.12110)
- [CoR-GS 官方代码](https://github.com/jiaw-z/CoR-GS)

## 4. 训练视角定量对比

以下 PSNR 是在 5 个已注册训练视角上的结果，仅用于检查模型对输入图像的拟合情况，不等同于未见测试视角的泛化指标。

| 视角 | FSGS PSNR (dB) | CoR-GS PSNR (dB) |
|---|---:|---:|
| `view_001` | 23.14 | 31.83 |
| `view_002` | 26.14 | 33.26 |
| `view_004` | 26.15 | 31.40 |
| `view_005` | 25.35 | 30.55 |
| `view_008` | 24.28 | 33.10 |
| 平均 | **25.01** | **32.03** |

CoR-GS 在本次训练视角上的 RGB 拟合明显高于之前的 FSGS 结果；但由于两次训练的迭代数、模型结构和深度正则状态不同，不能将该差异直接视为严格的算法优劣结论。

## 5. 结果解释

### FSGS

FSGS 主要针对少视图造成的初始 SfM 点云稀疏和空间覆盖不足，通过在相距较远的 Gaussian 之间补点来扩展场景表示，并可利用单目深度和伪视角进行几何约束。

### CoR-GS

CoR-GS 同时训练两个 Gaussian 场：

- 如果两个场的 Gaussian 位置差异很大，则通过 Co-Pruning 删除不一致点；
- 如果两个场在伪视角下的渲染差异很大，则通过 Co-Regularization 让它们相互约束。

因此，CoR-GS 主要通过“两个独立模型之间的一致性”抑制少视图下的错误几何和漂浮点。

## 6. 当前限制与下一步

- 8 张照片中只有 5 张被 COLMAP 成功注册，视角覆盖仍然不足。
- FSGS 本次没有启用真正的 MiDaS 深度正则化。
- 当前 PSNR 是训练视角指标，尚未进行严格的独立测试视角评估。
- 如果希望公平比较，应使用同一组已注册视角、相同训练迭代数，并分别运行 vanilla 3DGS、FSGS、CoR-GS 和 CoR-FSGS。

## 7. 训练暂停与异常暂存

已在 CoR-GS 训练脚本中加入可恢复暂存机制：

- 默认每 500 次迭代原子保存一次 `pause_latest.pth`；
- 暂存内容包括 `gs0`、`gs1`、Gaussian 参数、当前 SH 阶数、稠密化统计量和 Adam 优化器状态；
- 在训练终端按 `Ctrl+C` 时，会在当前迭代结束后保存并正常退出；
- 未捕获异常退出时，会尝试保存最近一个安全迭代的暂存文件；
- 写入采用临时文件替换，训练中断不会覆盖掉上一份完整暂存。

从暂存点继续训练时，使用 `--start_checkpoint`，并将 `--iterations` 设置为新的总迭代数。例如：

```powershell
$env:PYTHONPATH = "C:\\mine\\git\\SparseView3DGS\\program\\corgs;C:\\mine\\git\\SparseView3DGS\\program\\fsgs\\submodules\\diff-gaussian-rasterization-confidence;C:\\mine\\git\\SparseView3DGS\\program\\fsgs\\submodules\\simple-knn"
& "C:\\Users\\leo\\.conda\\envs\\ai\\python.exe" train.py `
  --source_path "C:\\mine\\git\\SparseView3DGS\\data\\fsgs\\latest_8_views" `
  --model_path "output\\cor_gs_resume" `
  --start_checkpoint "output\\cor_gs_run\\pause_latest.pth" `
  --iterations 5000 `
  --gaussiansN 2 --coreg --coprune
```

如需更频繁暂存，可增加 `--pause_checkpoint_interval 100`；设置为 `0` 可关闭周期暂存。该功能不替换原有的 `chkpnt*.pth`，旧版单 Gaussian 检查点仍可读取。

## 8. CoR-GS + Binocular Stereo Consistency 续训结果

### 训练过程

本次从已完成的 CoR-GS 3000 次结果继续训练，并同时恢复两个 Gaussian 场：GS0 使用原有 `chkpnt3000.pth` 及优化器状态，GS1 使用原有第二场 PLY 初始化。随后在 3000—4000 次迭代启用双目一致性约束。

训练配置：

```text
--gaussiansN 2
--coreg --coprune --coprune_interval 500
--binocular --binocular_start 3000
--binocular_interval 20 --binocular_weight 0.2
--densify_until_iter 3000
--pause_checkpoint_interval 250
```

中途出现了两类恢复兼容问题，均由自动暂存机制保留了可恢复状态：

1. 旧检查点恢复后，confidence 缓冲仍为初始 80 点，已改为按恢复后的 Gaussian 数量重建；
2. Co-Pruning 的临时 NumPy 索引被错误纳入异常暂存，已限制暂存器只保存具有 `capture()` 的 Gaussian 模型。

修复后从第 3250 次暂存恢复，顺利通过第 3500 次 Co-Pruning，并完成到第 4000 次迭代。

### 最终结果

- 输出目录：[latest_8_cor_gs_binocular_resume_4000](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_binocular_resume_4000)
- GS0 点云：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_binocular_resume_4000/point_cloud/iteration_4000/point_cloud.ply)，133,820 个 Gaussian
- GS1 点云：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_binocular_resume_4000/point_cloud_gs2/iteration_4000/point_cloud.ply)，134,604 个 Gaussian
- 可恢复暂存：[pause_latest.pth](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_binocular_resume_4000/pause_latest.pth)，记录第 4000 次迭代
- GS0 训练视图：[view_001.png](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_binocular_resume_4000/train/ours_4000/renders/view_001.png)
- GS0 深度图：[view_001_depth.png](C:/mine/git/SparseView3DGS/output/corgs/latest_8_cor_gs_binocular_resume_4000/train/ours_4000/renders/view_001_depth.png)

5 个已注册训练视图上的 PSNR：

| 视图 | PSNR (dB) |
|---|---:|
| `view_001` | 33.13 |
| `view_002` | 34.94 |
| `view_004` | 32.51 |
| `view_005` | 34.03 |
| `view_008` | 32.68 |
| 平均 | **33.46** |

该 PSNR 是训练视图拟合指标；由于当前 COLMAP 只注册了 5 个视图，尚未形成独立的新视角测试集，因此不能直接作为泛化性能结论。
## 9. 追加 9 张照片后的重建与离群视角修复（17 → 16 张）

### 数据与 COLMAP

- 原始输入共 17 张照片；由于 Python 端对中文文件名的读取存在编码问题，先复制为 ASCII 文件名 `view_001.jpg` … `view_017.jpg`。
- COLMAP 初次注册结果为 17/17，但 `view_003.jpg` 被估计到约 90,000 的异常相机位置，导致场景尺度从正常的约 9 变为 93851.62。
- 该异常 SfM 结果会使 Gaussian 与相机不对齐，表现为训练 loss 长期约 0.55、渲染全黑，17 图版本 PSNR 仅约 5.964 dB。
- 剔除这个 COLMAP 离群注册视角后重新建图：16/16 注册，1794 个稀疏点，`cameras_extent = 8.885362`。
- 不同图片分辨率使用 `single_camera: false`；训练输入由 COLMAP undistorter 生成。

### 修复后的训练配置

```text
--iterations 4000
--gaussiansN 2
--coreg --coprune --coprune_interval 500
--densify_until_iter 2500
--sample_pseudo_interval 10
--start_sample_pseudo 1500 --end_sample_pseudo 4000
--binocular --binocular_start 2500
--binocular_interval 20 --binocular_weight 0.2
--binocular_baseline_min 0.01 --binocular_baseline_max 0.08
--pause_checkpoint_interval 250
```

双目基线按 `baseline_fraction × cameras_extent` 计算，以避免 COLMAP 任意全局尺度导致固定世界坐标基线失效。

### 最终结果

- 训练输出：[latest_16_cor_gs_binocular_scaleaware_4000](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_4000)
- GS0 点云：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_4000/point_cloud/iteration_4000/point_cloud.ply)，136212 个 Gaussian
- GS1 点云：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_4000/point_cloud_gs2/iteration_4000/point_cloud.ply)，137609 个 Gaussian
- 可恢复暂停文件：[pause_latest.pth](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_4000/pause_latest.pth)，记录第 4000 次迭代
- 示例 RGB 渲染：[view_001.png](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_4000/train/ours_4000/renders/view_001.png)
- 示例深度渲染：[view_001_depth.png](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_4000/train/ours_4000/renders/view_001_depth.png)

16 个有效训练视角的 PSNR：

| 视图 | PSNR (dB) |
|---|---:|
| `view_001` | 23.718 |
| `view_002` | 26.076 |
| `view_004` | 23.919 |
| `view_005` | 20.437 |
| `view_006` | 27.206 |
| `view_007` | 30.467 |
| `view_008` | 22.386 |
| `view_009` | 29.885 |
| `view_010` | 29.615 |
| `view_011` | 29.170 |
| `view_012` | 29.562 |
| `view_013` | 30.413 |
| `view_014` | 29.665 |
| `view_015` | 31.799 |
| `view_016` | 28.396 |
| `view_017` | 28.601 |
| 平均 | **27.582** |

该 PSNR 是训练视角拟合指标，不代表独立新视角泛化性能。当前结果已解决异常相机造成的全黑渲染，但仍存在重影和模糊；下一步应优先改善拍摄覆盖、剔除低重叠/强反光图片，并考虑更严格的 COLMAP 几何验证。
## 10. 剔除 `view_003` 后的独立重训

为确认剔除异常 COLMAP 视角后结果可复现，使用相同的 16 张有效图片和相同训练参数，从头重新训练到 4000 次；原有 `latest_16_cor_gs_binocular_scaleaware_4000` 结果保留未覆盖。

- 重训输出：[latest_16_cor_gs_binocular_scaleaware_rerun_4000](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_rerun_4000)
- 输入：16 个有效视角，不包含异常 `view_003`
- `cameras_extent`：8.885362
- GS0 Gaussian：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_rerun_4000/point_cloud/iteration_4000/point_cloud.ply)，138571 个
- GS1 Gaussian：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_rerun_4000/point_cloud_gs2/iteration_4000/point_cloud.ply)，134337 个
- 暂停检查点：[pause_latest.pth](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_rerun_4000/pause_latest.pth)，记录第 4000 次迭代
- 示例渲染：[view_001.png](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_scaleaware_rerun_4000/train/ours_4000/renders/view_001.png)

16 个训练视角的平均 PSNR 为 **26.997 dB**（最低 19.050 dB，最高 31.687 dB）。上一轮相同配置为 27.582 dB，差异约 0.6 dB，属于 Gaussian 随机初始化和训练采样造成的波动；两次均已解决异常相机导致的全黑渲染问题。

## 11. 统一分辨率与共享内参后的重新训练

### 数据与相机处理

- 排除已确认的 COLMAP 异常视角 `view_003`，其余 16 张图全部缩放到统一的 1600×900。
- 使用 `config_latest_16_single.yaml`，设置 `single_camera: true`，让 COLMAP 为所有图像估计并共享同一个相机内参模型。
- COLMAP 生成了两个 sparse 模型：`sparse/0` 仅注册 4 张图，`sparse/1` 注册完整 16 张图并生成 1751 个稀疏点，因此本次选用 `sparse/1` 做 undistortion 和训练输入。
- `sparse/1` 的共享 OPENCV 相机参数为 `[fx=1191.147, fy=1190.874, cx=800, cy=450, k1=0.008, k2=-0.048, p1=0.002, p2=-0.001]`。

### 训练配置

```text
--iterations 4000
--gaussiansN 2
--coreg --coprune --coprune_interval 500
--densify_until_iter 2500
--sample_pseudo_interval 10
--start_sample_pseudo 1500 --end_sample_pseudo 4000
--binocular --binocular_start 2500
--binocular_interval 20 --binocular_weight 0.2
--binocular_baseline_min 0.01 --binocular_baseline_max 0.08
--pause_checkpoint_interval 250
```

### 结果

- 输出目录：[latest_16_cor_gs_binocular_uniform_single_4000](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_uniform_single_4000)
- GS0：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_uniform_single_4000/point_cloud/iteration_4000/point_cloud.ply)，145,663 个 Gaussian
- GS1：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_uniform_single_4000/point_cloud_gs2/iteration_4000/point_cloud.ply)，148,929 个 Gaussian
- 暂存点：[pause_latest.pth](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_uniform_single_4000/pause_latest.pth)，记录第 4000 次迭代
- RGB 渲染：[view_001.png](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_uniform_single_4000/train/ours_4000/renders/view_001.png)
- 深度渲染：[view_001_depth.png](C:/mine/git/SparseView3DGS/output/corgs/latest_16_cor_gs_binocular_uniform_single_4000/train/ours_4000/renders/view_001_depth.png)

16 个训练视角的 PSNR：

| 视图 | PSNR (dB) |
|---|---:|
| `view_001` | 22.542 |
| `view_002` | 25.830 |
| `view_004` | 20.184 |
| `view_005` | 23.871 |
| `view_006` | 28.843 |
| `view_007` | 31.505 |
| `view_008` | 23.569 |
| `view_009` | 30.005 |
| `view_010` | 31.354 |
| `view_011` | 31.058 |
| `view_012` | 31.168 |
| `view_013` | 31.082 |
| `view_014` | 31.086 |
| `view_015` | 33.265 |
| `view_016` | 30.509 |
| `view_017` | 30.488 |
| 平均 | **28.523** |

### 分析

相对于上一轮“每张图独立内参”的 16 视角重训结果（平均 26.997 dB），统一分辨率并共享内参后提高约 **1.526 dB**；相对于同配置的另一轮独立内参训练（27.582 dB），提高约 **0.941 dB**。这说明前一轮的主要问题确实部分来自逐图估计出的焦距差异过大，而不是单纯由 Gaussian 数量不足造成。

从 RGB 渲染看，键盘、杯子和桌垫的轮廓已经能够重建，但仍有明显重影和模糊。统一内参改善了相机投影的一致性，却不能修复 COLMAP 位姿误差、运动/反光区域和视角覆盖不足；当前 PSNR 仍是训练视角拟合指标，不等同于新视角泛化质量。

注意：`prepare.py` 当前会优先选 `sparse/0`，而本次完整模型位于 `sparse/1`。后续若自动化重跑，应改为按注册图像数选择最大的 sparse 模型。

## 12. 统一内参实验中发现的新问题与解决方法

### 问题 1：逐图独立估计内参导致投影不一致

在此前的 16 图重建中，COLMAP 为每张图片建立了独立相机。不同图片得到的焦距差异较大，例如部分相机的 `fx/fy` 约为 1250，而另一些相机的焦距超过 2000。对于同一台相机连续拍摄的图片，这种差异通常不符合实际，会使 Gaussian 在不同视角之间无法稳定对齐，表现为重影、拉伸和模糊。

解决方法：

- 将相机设置为共享模型：`single_camera: true`；
- 使用统一的图像尺寸 1600×900 后重新运行 COLMAP；
- 让所有图片使用同一组焦距、主点和畸变参数。

效果：平均训练视角 PSNR 从 26.997 dB 提升到 28.523 dB，说明内参一致性确实是上一轮质量下降的重要原因之一。

### 问题 2：输入图片分辨率不一致

新增图片中存在尺寸差异。若直接使用 `single_camera: true`，同一个相机模型无法同时正确解释不同宽高的像素坐标，容易导致主点和焦距的尺度关系不一致，也会影响 COLMAP 的匹配和后续 undistortion。

解决方法：

- 保留原始图片不变；
- 另建统一输入目录，将 16 张有效图片缩放为 1600×900；
- 仅使用统一尺寸目录进行 COLMAP、undistortion 和 Gaussian 训练。

### 问题 3：COLMAP 默认选择了不完整的 sparse 模型

本次 COLMAP 输出了多个模型：`sparse/0` 只注册了 4 张图片，而 `sparse/1` 注册了全部 16 张图片。`prepare.py` 当前默认优先使用 `sparse/0`，会导致训练输入只包含少量视角，结果与预期不符。

解决方法：

- 检查每个 sparse 模型实际注册的图片数量；
- 手动选择注册图片最多的 `sparse/1`；
- 使用该模型完成 undistortion，并生成包含 1751 个稀疏点的训练输入。

后续自动化处理应修改为“选择注册图片数最多的 sparse 模型”，而不是固定选择 `sparse/0`。

### 问题 4：异常视角会破坏场景尺度

`view_003` 的 COLMAP 相机中心异常，位置约为 90,000，导致场景范围从正常的约 9 膨胀到 93851.62。此时 Binocular Stereo Consistency 使用的基线约束也会失效，Gaussian 与相机位置不再匹配，最终出现全黑或严重错误的渲染。

解决方法：

- 根据相机中心离群程度和场景尺度检测异常视角；
- 从 SfM 和训练输入中排除 `view_003`；
- 保留原始图片文件，不进行物理删除；
- 使用剩余 16 张图片重新进行 COLMAP 和训练。

### 问题 5：统一内参后仍存在重影和模糊

统一内参只能改善相机投影模型的一致性，不能自动修复以下问题：

- COLMAP 相机位姿仍可能存在误差；
- 键盘、杯子等物体的有效视角覆盖不足；
- 桌面和屏幕存在反光、遮挡或重复纹理；
- 图片之间可能有较大的视角跳变；
- 当前 PSNR 是训练视角拟合指标，不能代表新视角质量。

解决方法与后续方向：

- 继续使用异常相机检测和最少注册视角检查；
- 优先补拍相邻视角，避免只增加大跨度视角；
- 剔除低重叠、严重模糊、强反光或动态物体图片；
- 对 COLMAP 的相机轨迹和重投影误差进行筛选；
- 使用独立测试视角评估，而不只比较训练视角 PSNR；
- 在确认相机几何可靠后，再调整 Gaussian 数量、正则化强度和 Binocular 权重。

### 本轮结论

本轮问题的主要解决顺序为：

```text
异常视角剔除
    → 输入分辨率统一
    → 共享相机内参
    → 选择注册数最多的 COLMAP 模型
    → CoR-GS + Binocular Stereo Consistency 训练
```

统一内参后结果有定量改善，但当前瓶颈已经从“相机内参不一致”转移到“相机位姿精度、视角覆盖和场景非朗伯特性”。

## 13. 新增照片分析与重新训练结果

### 数据分析

本轮 `data` 目录共包含 35 张图片：

- 33 张为 1920×1080，2 张为 1440×810；
- 已建立统一的 1600×900 副本，原始图片未删除；
- 基于 Laplacian 方差的清晰度指标约为 58.6–190.8，整体没有明显不可用的模糊图；
- 新增照片对应 `view_018`–`view_035`，视角更连续、覆盖更充分，且多数图片清晰度良好。

### 全量 35 图 COLMAP 结果

35 张图全部注册成功，生成 7,657 个稀疏点。平均重投影误差约 0.744 px，95% 分位约 1.663 px，没有超过 4 px 的稀疏点。但相机中心在不同拍摄批次之间存在较大跳变，最大相邻跳变约 9.9 个场景单位，说明“注册成功”并不等于全局轨迹完全一致。

### 全量 35 图训练结果

使用统一分辨率、共享 OPENCV 内参以及与上一轮相同的 CoR-GS + Binocular 配置训练 4000 次：

- 输出目录：[latest_35_cor_gs_binocular_uniform_single_4000](C:/mine/git/SparseView3DGS/output/corgs/latest_35_cor_gs_binocular_uniform_single_4000)
- GS0 Gaussian：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_35_cor_gs_binocular_uniform_single_4000/point_cloud/iteration_4000/point_cloud.ply)，228,248 个
- GS1 Gaussian：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_35_cor_gs_binocular_uniform_single_4000/point_cloud_gs2/iteration_4000/point_cloud.ply)，230,685 个
- 暂停点：[pause_latest.pth](C:/mine/git/SparseView3DGS/output/corgs/latest_35_cor_gs_binocular_uniform_single_4000/pause_latest.pth)
- RGB 渲染：[view_003.png](C:/mine/git/SparseView3DGS/output/corgs/latest_35_cor_gs_binocular_uniform_single_4000/train/ours_4000/renders/view_003.png)
- 35 个训练视角平均 PSNR：**25.867 dB**
- 已知早期 16 张有效图（排除 `view_003`）在该模型上的平均 PSNR：**22.720 dB**
- 最差视角为 `view_003`：12.072 dB，其次为 `view_005`、`view_004` 和 `view_008`。

这说明将不同拍摄批次的 35 张图片直接混合，虽然 COLMAP 可以得到低重投影误差的模型，但 Gaussian 训练仍受到相机轨迹跨度、遮挡和视角分布不一致的影响。

### 新增 18 图独立重建与训练

为验证新增照片本身的质量，单独使用 `view_018`–`view_035` 重新运行 COLMAP 和训练：

- 18/18 张图片注册成功；
- 生成 3,989 个稀疏点；
- 使用同样的统一分辨率、共享内参和 4000 次 CoR-GS + Binocular 训练；
- 输出目录：[latest_18_new_cor_gs_binocular_uniform_single_4000](C:/mine/git/SparseView3DGS/output/corgs/latest_18_new_cor_gs_binocular_uniform_single_4000)
- GS0 Gaussian：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_18_new_cor_gs_binocular_uniform_single_4000/point_cloud/iteration_4000/point_cloud.ply)，229,347 个
- GS1 Gaussian：[point_cloud.ply](C:/mine/git/SparseView3DGS/output/corgs/latest_18_new_cor_gs_binocular_uniform_single_4000/point_cloud_gs2/iteration_4000/point_cloud.ply)，230,869 个
- 暂存点：[pause_latest.pth](C:/mine/git/SparseView3DGS/output/corgs/latest_18_new_cor_gs_binocular_uniform_single_4000/pause_latest.pth)
- RGB 渲染：[view_018.png](C:/mine/git/SparseView3DGS/output/corgs/latest_18_new_cor_gs_binocular_uniform_single_4000/train/ours_4000/renders/view_018.png)
- 18 个训练视角平均 PSNR：**33.021 dB**
- 最低 PSNR：30.746 dB；最高 PSNR：34.990 dB。

### 结论

新增照片是有效的，问题主要出在不同拍摄批次之间的联合建图和联合训练，而不是新增照片质量差。当前最好的结果是新增 18 图独立模型，平均 PSNR 33.021 dB，接近之前 5 个注册视角上的 33.46 dB，同时覆盖了更多视角。

因此后续应优先采用“同一拍摄批次内建图”的策略；如果必须合并早期和新增照片，应先对相机轨迹进行尺度、位姿连续性和重投影一致性筛选，再进行联合训练。
