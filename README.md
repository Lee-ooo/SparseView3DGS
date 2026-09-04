# SparseView3DGS

## 目录

```text
SparseView3DGS/
├─ data/                         # 原始照片和数据集
├─ train/                        # 全部几何、训练程序和后端
│  ├─ sfm/                       # COLMAP、PDCNet+ 和 SfM 整理
│  ├─ corgs/                     # CoR-GS 后端及复用的 CUDA 扩展
│  ├─ binocular/                 # Binocular 一致性实现
│  └─ run_pipeline.py            # 分阶段入口
├─ render/                       # RGB/深度/交互式渲染入口
├─ output/                       # 重建、模型、检查点和渲染结果
└─ config.yaml                   # COLMAP/SfM 配置
```

## 总体流程

```text
原始照片
   │
   ▼
SfM / COLMAP
   ├─ cameras.bin    相机内参
   ├─ images.bin     相机位姿
   ├─ points3D.bin   COLMAP 稀疏点云（保留，不作为最终初始化）
   └─ images/        去畸变图像
   │
   ▼
PDCNet+
   └─ points3D.ply   稠密匹配三角化点云（替换初始化）
   │
   ▼
CoR-GS + Binocular
   └─ 唯一保留的 Gaussian 训练方法
   │
   ▼
Render
   └─ RGB、深度图和自由视角
```

## 命令

从项目根目录执行：

```powershell
cd C:\mine\git\SparseView3DGS

# COLMAP + PDCNet+（包含相机、去畸变和稠密初始化）
python .\train\run_pipeline.py --stage sfm

# CoR-GS + Binocular 训练
python .\train\run_pipeline.py --stage train

# 指定视角渲染 RGB 和深度
python .\render\render_views.py --views view_018,view_026,view_035 --render-depth
```

## 分阶段命令

```powershell
# 图片分析
python .\train\run_pipeline.py --stage analyze --images .\data --workspace .\output\reconstruction\analysis_latest

# COLMAP SfM + PDCNet+ 稠密初始化
python .\train\run_pipeline.py --stage sfm --images .\data --workspace .\output\reconstruction\new_reconstruction --config .\config.yaml

# 已有 COLMAP 数据时只重跑 PDCNet+
python .\train\run_pipeline.py --stage pdcnet --dataset .\output\reconstruction\latest_18_new_recon_uniform_single\dense

# CoR-GS + Binocular 训练
python .\train\run_pipeline.py --stage train --dataset .\output\reconstruction\latest_18_new_recon_uniform_single\dense --model-path .\output\corgs\latest_18_new_cor_gs_binocular_uniform_single_4000 --iterations 4000

# 断点恢复
python .\train\run_pipeline.py --stage resume --dataset .\output\reconstruction\latest_18_new_recon_uniform_single\dense --model-path .\output\corgs\corgs_resume_6000 --start-checkpoint .\output\corgs\latest_18_new_cor_gs_binocular_uniform_single_4000\pause_latest.pth --iterations 6000
```

## 数据与输出

输入数据位于 `data/`。训练输入中的关键文件为：

- `cameras.bin`：COLMAP 相机内参；
- `images.bin`：COLMAP 相机位姿；
- `points3D.bin`：COLMAP 原始稀疏点云，保留作备份；
- `points3D.ply`：PDCNet+ 稠密初始化点云，优先读取。

```text
output/reconstruction/<name>/
├─ database.db
├─ sparse/                         # COLMAP 原始模型
├─ dense/
│  ├─ images/                      # 去畸变图像
│  └─ sparse/0/
│     ├─ cameras.bin              # 保留的 COLMAP 内参
│     ├─ images.bin               # 保留的 COLMAP 位姿
│     ├─ points3D.bin             # COLMAP 稀疏点云备份
│     └─ points3D.ply             # PDCNet+ 稠密初始化
├─ pdcnet/                        # PDCNet+ 中间输出
└─ manifest.json
```

## 渲染与交互查看

渲染程序位于 `render/`，后端固定为 CoR-GS + Binocular。渲染不会重新运行 SfM 或 PDCNet+，只读取已经训练好的模型和 COLMAP 相机数据。

### 自由视角

直接运行以下命令即可启动自由视角交互界面：

```powershell
python .\render\render_views.py
```

如需覆盖默认数据集、模型或迭代次数，可继续追加 `--dataset`、`--model-path` 和 `--iteration`。

鼠标左键旋转，右键平移，滚轮缩放；`R` 重置，`S` 保存 RGB 和位姿 JSON，`Q` 或 `Esc` 退出。结果保存到模型目录下的 `interactive/`。

### 指定视角和深度图

```powershell
python .\render\render_views.py `
  --dataset .\output\reconstruction\latest_18_new_recon_uniform_single\dense `
  --model-path .\output\corgs\latest_18_new_cor_gs_binocular_uniform_single_4000 `
  --iteration 4000 `
  --views view_018,view_026,view_035 `
  --render-depth
```

训练前请确认数据集中的 `sparse/0/points3D.ply` 是 PDCNet+ 生成的稠密初始化点云。
