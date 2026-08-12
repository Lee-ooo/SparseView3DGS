# SparseView3DGS 在另一台 Windows 电脑上的配置指南

本文用于从 GitHub 重新克隆项目，并在一台没有现成环境的 Windows 电脑上完成配置。

项目分为两个阶段：

1. COLMAP：根据照片估计相机位姿和稀疏点云。
2. FSGS/CoR-GS：读取 COLMAP 生成的数据，训练和渲染 Gaussian Splatting 模型。

不要把 COLMAP 和 Gaussian Splatting 当成同一个程序安装或运行。

## 1. 硬件和软件要求

建议配置：

- Windows 10/11 64 位
- NVIDIA GPU，建议显存至少 8 GB；RTX 4060 已验证
- 最新的 NVIDIA 显卡驱动
- Git
- Miniconda 或 Anaconda
- Visual Studio Build Tools（包含“使用 C++ 的桌面开发”和 Windows SDK）
- CUDA Toolkit 13.0，需提供 `nvcc.exe`
- COLMAP Windows CUDA 版本

项目当前验证的 Python/PyTorch 组合为：

```text
Python 3.11
PyTorch 2.11.0+cu130
torchvision 与 PyTorch 2.11 对应的版本
```

CUDA Toolkit 主要用于编译项目的 CUDA 扩展；PyTorch 的 CUDA wheel 自带运行时。两者都需要准备，不能只安装 Python 包。

## 2. 克隆项目

在目标位置打开终端：

```powershell
git clone https://github.com/Lee-ooo/SparseView3DGS.git
Set-Location .\SparseView3DGS
```

项目仓库不包含以下本地文件，需要在新电脑上重新准备：

- 原始照片和训练数据
- `output/`、`model/` 下的模型、检查点和渲染结果
- CUDA 编译生成的 `.pyd` 文件
- PyTorch wheel、MiDaS 等大型权重
- COLMAP 可执行文件

这些文件被 `.gitignore` 排除，避免把数据和数百 MB/GB 的二进制文件提交到 GitHub。

## 3. 创建 Conda 环境

建议使用独立环境，不要直接使用系统 Python：

```powershell
conda create -n sparseview3dgs python=3.11 -y
conda activate sparseview3dgs
python --version
```

安装 PyTorch。推荐参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)；本项目当前机器使用的是 `2.11.0+cu130`。对应的 Windows/Pip/CUDA 13.0 命令为：

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

输出应显示 PyTorch 2.11、CUDA 可用，并显示你的 NVIDIA GPU。若 `torch.cuda.is_available()` 为 `False`，先解决显卡驱动或 PyTorch 安装问题，不要继续编译扩展。

安装项目侧的 Python 依赖：

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r .\requirements.txt
python -m pip install plyfile tqdm matplotlib torchmetrics==1.2.0 timm==0.9.12 imageio open3d
```

项目根目录的 `requirements.txt` 负责图像分析和配置读取；后端还需要上面列出的训练、点云和评估依赖。

## 4. 配置 CUDA 编译工具链

打开新的终端，确认以下命令可用：

```powershell
nvcc --version
where.exe cl
where.exe cmake
```

如果 `cl` 找不到，请从 Visual Studio Build Tools 的“x64 Native Tools Command Prompt”运行后续命令，或先运行对应的 `vcvars64.bat`。如果 `nvcc` 找不到，把 CUDA Toolkit 的 `bin` 目录加入 PATH。

如果电脑安装了多个 CUDA 版本，建议显式指定 CUDA 路径。例如：

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:CUDA_PATH = $env:CUDA_HOME
$env:Path = "$env:CUDA_HOME\bin;$env:Path"
```

重新打开终端后再次执行 `nvcc --version`，确认版本正确。

## 5. 准备 COLMAP

下载或复制 Windows CUDA 版 COLMAP，并放到：

```text
program/colmap-x64-windows-cuda/
```

至少应存在：

```text
program/colmap-x64-windows-cuda/bin/colmap.exe
```

验证：

```powershell
& .\program\colmap-x64-windows-cuda\bin\colmap.exe -h
```

也可以把 `colmap.exe` 加入 PATH，然后在所有命令中使用 `--colmap colmap`。默认脚本使用项目内的完整路径。

## 6. 编译 CUDA 扩展

仓库包含 CUDA 扩展的源代码，但不包含当前电脑生成的 `.pyd` 文件。首次配置必须编译：

```powershell
conda activate sparseview3dgs
Set-Location .\SparseView3DGS

python -m pip install -v .\program\fsgs\submodules\simple-knn
python -m pip install -v .\program\fsgs\submodules\diff-gaussian-rasterization-confidence
```

如果 Windows 编译器报编码或不支持的编译器版本错误，可以使用项目提供的包装脚本：

```powershell
Set-Location .\program\fsgs\submodules\diff-gaussian-rasterization-confidence
python ..\..\..\build_fsgs_extension.py build_ext --inplace
```

然后编译 `simple-knn`：

```powershell
Set-Location ..\simple-knn
python setup.py build_ext --inplace
Set-Location ..\..\..\..
```

编译成功后，应看到类似文件：

```text
program/fsgs/submodules/simple-knn/simple_knn/_C*.pyd
program/fsgs/submodules/diff-gaussian-rasterization-confidence/diff_gaussian_rasterization/_C*.pyd
```

CoR-GS 默认复用 FSGS 的这两个扩展。若某个后端目录下的扩展路径不同，优先保留 FSGS 扩展，并确认运行时的 `PYTHONPATH` 包含 `program/fsgs/submodules/`。

## 7. 准备图片目录

把同一物体或场景的照片放在项目根目录的 `data/` 下：

```text
SparseView3DGS/
└─ data/
   ├─ view_001.jpg
   ├─ view_002.jpg
   └─ view_003.jpg
```

建议：

- 小物体至少拍摄 8–12 张连续环绕照片。
- 相邻照片保持约 60%–80% 重叠和适度视差。
- 保持场景静止，尽量锁定曝光、白平衡、焦距和分辨率。
- 不要把多个场景或历史数据集混在 `data/` 根目录；默认只读取这一层的图片。
- 中文文件名可以使用，准备脚本会复制成 ASCII 文件名供 COLMAP 使用。

## 8. 冒烟测试

先只分析图片，不运行 COLMAP：

```powershell
python .\program\scripts\run_pipeline.py `
  --stage analyze `
  --images .\data `
  --workspace .\output\reconstruction\setup_check
```

检查报告：

```text
output/reconstruction/setup_check/reports/image_quality.json
```

然后验证项目 Python 模块和 CUDA 扩展：

```powershell
$env:PYTHONPATH = (Resolve-Path .\program).Path
python -c "import torch, simple_knn, diff_gaussian_rasterization; print('imports ok'); print(torch.cuda.is_available())"
```

如果这里失败，通常是 Python 环境、CUDA Toolkit、Visual Studio 编译器或 `.pyd` 路径不匹配。

## 9. 第一次完整运行

推荐先使用较少迭代验证流程：

```powershell
python .\program\scripts\run_pipeline.py `
  --stage train `
  --method corgs `
  --images .\data `
  --workspace .\output\reconstruction\first_run `
  --model-path .\output\corgs\first_run `
  --iterations 100
```

正常完成后，再进行正式训练：

```powershell
python .\program\scripts\run_pipeline.py `
  --stage train `
  --method corgs `
  --images .\data `
  --workspace .\output\reconstruction\recon_latest `
  --model-path .\output\corgs\corgs_latest `
  --iterations 4000
```

常用方法：

```text
fsgs        FSGS
corgs       CoR-GS + Binocular
corgs_fsgs  FSGS + CoR-GS + Binocular，默认组合
```

## 10. 断点恢复和渲染

恢复训练时使用新的模型目录，不覆盖旧实验：

```powershell
python .\program\scripts\run_pipeline.py `
  --stage resume `
  --method corgs `
  --dataset .\output\reconstruction\recon_latest\dense `
  --model-path .\output\corgs\resume_6000 `
  --start-checkpoint .\output\corgs\corgs_latest\pause_latest.pth `
  --iterations 6000
```

指定视角渲染：

```powershell
python .\program\scripts\run_pipeline.py `
  --stage render `
  --method corgs `
  --dataset .\output\reconstruction\recon_latest\dense `
  --model-path .\output\corgs\corgs_latest `
  --iterations 4000 `
  --views view_018,view_026,view_035 `
  --render-depth
```

交互式自由视角：

```powershell
python .\program\scripts\run_pipeline.py `
  --stage interactive `
  --method corgs `
  --dataset .\output\reconstruction\recon_latest\dense `
  --model-path .\output\corgs\corgs_latest `
  --iterations 4000
```

## 11. 常见问题

### `torch.cuda.is_available()` 为 `False`

检查 NVIDIA 驱动、PyTorch CUDA wheel 和当前 Conda 环境。确认命令行中的 `python` 就是目标环境的 Python：

```powershell
where.exe python
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

### `nvcc` 或 `cl` 找不到

这是编译工具链问题，不是项目 Python 代码问题。安装 CUDA Toolkit 和 Visual Studio C++ Build Tools，并从具有正确环境变量的终端重新编译。

### `No module named simple_knn` 或 `diff_gaussian_rasterization`

重新编译扩展，并确认 `.pyd` 文件存在。运行项目脚本时不要删除它设置的 `PYTHONPATH`；项目脚本会自动加入 FSGS/CoR-GS 和扩展目录。

### COLMAP 找不到或注册图片太少

确认 `program/colmap-x64-windows-cuda/bin/colmap.exe` 存在，并查看：

```text
output/reconstruction/<name>/manifest.json
output/reconstruction/<name>/reports/image_quality.json
```

如果注册图片不足，优先改进拍摄重叠、纹理和光照，不要先调整 Gaussian 参数。

### 训练输出为空或渲染全黑

先确认 `dense/images/` 和 `dense/sparse/0/` 存在，再确认模型目录包含点云和检查点。检查 GPU、扩展导入和训练日志；不要把训练视角 PSNR 当作独立新视角质量。

## 12. 相关文档

- [`README_ZH.md`](README_ZH.md)：项目简介和常用入口
- [`PIPELINE_ZH.md`](PIPELINE_ZH.md)：完整流程、参数和实验记录
- [`VIEWER_USAGE_ZH.md`](VIEWER_USAGE_ZH.md)：交互式查看器说明
- [`training_results.md`](training_results.md)：历史训练结果
