# SparseView3DGS Agent Guide

## 项目目标

本项目使用少量照片完成 COLMAP/SfM + Gaussian Splatting 三维重建，重点实验包括 FSGS、CoR-GS、Binocular Stereo Consistency、异常视角筛选、断点恢复和交互式 Novel View 渲染。

COLMAP 负责相机位姿和稀疏点云；FSGS/CoR-GS 负责 Gaussian 参数训练和渲染。不要把 COLMAP 的 SfM 与 Gaussian Splatting 训练混为一个阶段。

## 固定环境

- Windows；命令入口全部使用 Python，不保留 PowerShell 脚本；
- Conda 环境：`ai`；
- Python：`C:\Users\leo\.conda\envs\ai\python.exe`；
- PyTorch：`2.11.0+cu130`；
- GPU：NVIDIA GeForce RTX 4060；
- COLMAP：`program/colmap-x64-windows-cuda/bin/colmap.exe`；
- 已编译 CUDA 扩展：`program/fsgs/submodules/`。

优先复用 `ai` 环境和现有扩展，不要创建新环境或重复下载软件包。CoR-GS 默认复用 FSGS 已编译的 `simple-knn` 和 `diff-gaussian-rasterization-confidence`。

## 根目录组织

根目录只把工作内容分为三个主要文件夹：

```text
SparseView3DGS/
├─ data/                    # 原始照片和训练数据
│  ├─ *.jpg                 # 当前原始输入照片
│  ├─ fsgs/                 # FSGS 可直接读取的历史数据集
│  └─ processed/            # 统一分辨率等处理后的照片副本
├─ program/                 # 所有程序和依赖软件
│  ├─ fsgs/                 # FSGS 后端和 CUDA 扩展
│  ├─ corgs/                # CoR-GS 后端和 CUDA 扩展
│  ├─ colmap-x64-windows-cuda/
│  ├─ sparseview3dgs/       # 项目 Python 工具
│  └─ scripts/              # 当前保留的 Python 入口
└─ output/                  # 所有重建、模型、检查点和渲染结果
   ├─ reconstruction/       # COLMAP 中间结果和后端数据
   ├─ fsgs/                 # FSGS 训练输出
   └─ corgs/                # CoR-GS 训练输出
```

旧版 PowerShell 入口已全部删除。当前只使用：

- `program/scripts/run_pipeline.py`：分析、COLMAP、训练、续训、渲染和交互查看；
- `program/scripts/render_views.py`：指定视角渲染。

## 后端数据格式

FSGS/CoR-GS 的 `Dataset` 必须包含：

```text
dataset/
├─ images/
└─ sparse/
   └─ 0/
      ├─ cameras.bin
      ├─ images.bin
      └─ points3D.bin
```

`program/sparseview3dgs/prepare.py` 会运行 COLMAP、自动选择注册图片最多的 sparse 模型，并将去畸变输出整理为 `dense/sparse/0`。

## 标准工作流

从项目根目录执行：

```text
cd C:\mine\git\SparseView3DGS
```

### 图片分析

```text
python .\program\scripts\run_pipeline.py --stage analyze --images .\data --workspace .\output\reconstruction\analysis_latest
```

报告：`output/reconstruction/analysis_latest/reports/image_quality.json`。

### COLMAP 预处理

```text
python .\program\scripts\run_pipeline.py --stage prepare --images .\data --workspace .\output\reconstruction\recon_latest
```

主要输出：`output/reconstruction/recon_latest/database.db`、`sparse/`、`dense/`、`manifest.json`。

### FSGS 训练

```text
python .\program\scripts\run_pipeline.py --stage train --method fsgs --dataset .\output\reconstruction\recon_latest\dense --model-path .\output\fsgs\fsgs_latest --iterations 4000
```

### CoR-GS + Binocular 训练

```text
python .\program\scripts\run_pipeline.py --stage train --method corgs --dataset .\output\reconstruction\recon_latest\dense --model-path .\output\corgs\corgs_latest --iterations 4000
```

默认启用 `--gaussiansN 2`、`--coreg`、`--coprune`、Binocular、伪视角一致性和周期性暂停检查点。

### 断点恢复

```text
python .\program\scripts\run_pipeline.py --stage resume --method corgs --dataset .\output\reconstruction\recon_latest\dense --model-path .\output\corgs\corgs_resume_6000 --start-checkpoint .\output\corgs\corgs_latest\pause_latest.pth --iterations 6000
```

恢复训练使用新的 `ModelPath`，不要覆盖原实验。

### 指定视角渲染

```text
python .\program\scripts\render_views.py --method corgs --dataset .\data\fsgs\latest_18_new_uniform_single_views --model-path .\output\corgs\latest_18_new_cor_gs_binocular_uniform_single_4000 --iteration 4000 --views view_018,view_026,view_035 --render-depth
```

### 交互式自由视角

```text
python .\program\scripts\run_pipeline.py --stage interactive --method corgs --dataset .\data\fsgs\latest_18_new_uniform_single_views --model-path .\output\corgs\latest_18_new_cor_gs_binocular_uniform_single_4000 --iterations 4000
```

鼠标左键旋转、右键平移、滚轮缩放；`S` 保存 RGB 图和位姿 JSON，`R` 重置，`Q/Esc` 退出。默认保存到 `<model>/interactive/`。

## 当前最佳结果

当前较好的实验是新增 18 张照片独立重建：

- 输入：`view_018`–`view_035`；
- 统一分辨率：约 `1600×900`；
- 共享 `OPENCV` 内参；
- 方法：CoR-GS + Binocular；
- 迭代：4000；
- 平均训练视角 PSNR：`33.021 dB`；
- 模型：`output/corgs/latest_18_new_cor_gs_binocular_uniform_single_4000`；
- GS0/GS1：约 229,347 / 230,869 个 Gaussian。

该 PSNR 只是训练视角拟合指标，不能替代独立新视角评估。

## 三张照片实验的事实

当前三视图 FSGS 实验并不是三张图片联合训练。`output/fsgs/three_views_fsgs_2000/cfg_args` 显示：

- 数据源是 `three_views_recon_pair12`；
- `n_views=2`；
- 训练输出只有 `view_001` 和 `view_002`；
- 最终点云约 10,366 个 Gaussian。

三张照片先进行成对 COLMAP 尝试，最终只有一对生成了可用相机和稀疏点云，第三张未注册照片不会自动参与 FSGS 训练。

## 少视图质量要求

- 小物体建议 8–12 张连续环绕照片；
- 相邻照片保持约 60%–80% 重叠和适度视差；
- 场景静止，曝光、白平衡、焦距和变焦稳定；
- 统一图片分辨率和宽高比，使用共享内参；
- COLMAP 尽量注册所有有效照片；
- 检查相机中心轨迹、重投影误差和稀疏点云覆盖范围；
- 先保证 SfM 几何可靠，再调 Gaussian 数量、正则化和训练轮数；
- 训练视角 PSNR 之外，必须检查留出视角或交互式新视角。

FSGS 能缓解初始点云过少，CoR-GS 能抑制少视图过拟合，但都不能修复错误的 COLMAP 位姿、异常场景尺度或缺失的真实视角。

## 修改规范

1. 先检查 `data/`、`program/` 和 `output/`，不要删除或覆盖已有实验。
2. 新实验使用独立的 `output/reconstruction/<name>` 和 `output/<method>/<name>`。
3. 优先使用固定的 `ai` Python，不创建新环境。
4. 修改 `PYTHONPATH` 时保留 `program/fsgs/submodules` 的已编译扩展路径。
5. 中文文件名优先使用 Pillow 读取，不要依赖 Windows 下 `cv2.imread`。
6. 修改 COLMAP 流程后检查 `manifest.json` 和 `dense/sparse/0`。
7. 训练或渲染后检查模型、点云、检查点和关键图片是否存在且不是全黑。
8. 修改后端代码后执行 Python 语法检查；渲染器改动应使用已有模型做最小验证。
9. 新实验和问题分析同步记录到 `training_results.md` 或 `PIPELINE_ZH.md`。

## 验证命令

```powershell
& C:\Users\leo\.conda\envs\ai\python.exe -m py_compile `
  program\sparseview3dgs\prepare.py `
  program\sparseview3dgs\interactive_viewer.py
```

```powershell
& C:\Users\leo\.conda\envs\ai\python.exe `
  -m sparseview3dgs.prepare `
  --images .\data `
  --workspace .\output\reconstruction\analysis_smoke `
  --config .\config.yaml `
  --analyze-only
```

运行模块时需设置：

```powershell
$env:PYTHONPATH = (Resolve-Path .\program).Path
```

## 相关文档

- [PIPELINE_ZH.md](C:/mine/git/SparseView3DGS/PIPELINE_ZH.md)：完整流程和命令；
- [training_results.md](C:/mine/git/SparseView3DGS/training_results.md)：历史实验结果；
- [README_ZH.md](C:/mine/git/SparseView3DGS/README_ZH.md)：项目简介。
