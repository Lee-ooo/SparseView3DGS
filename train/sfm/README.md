# SfM 与 PDCNet+ 稠密初始化

本目录把两个几何环节明确分开：

1. `prepare.py` 运行 COLMAP，生成相机内参、相机位姿和去畸变图像。
2. `pdcnet_init.py` 调用 Binocular3DGS 使用的官方 PDCNet+ `triangulate.py`，读取 COLMAP 的 `cameras.bin/images.bin`，将稠密三角化结果写成 `sparse/0/points3D.ply`。

因此 PDCNet+ 只替换稀疏点云初始化，不替换 COLMAP 的 SfM 位姿。原始 `points3D.bin` 会保留。

## 权重

官方 Binocular3DGS README 的权重链接：

https://drive.google.com/file/d/151X9ovbOG35tbPjioV5CYk_5GKQ8FErw/view?usp=sharing

下载后放到：

```text
train/sfm/dense_matcher/pre_trained_models/PDCNet_plus_megadepth.pth
```

也接受 `PDCNet_plus_megadepth.pth.tar`。执行：

```powershell
python .\train\run_pipeline.py --stage pdcnet --dataset .\output\reconstruction\latest_18_new_recon_uniform_single\dense
```

输出文件：`<dataset>\sparse\0\points3D.ply`。
