输入照片请放在 data/，然后从项目根目录执行：

```powershell
python .\train\run_pipeline.py --stage sfm --images .\data --workspace .\output\reconstruction\new_reconstruction
```

SfM 阶段保留 COLMAP 相机位姿，并用 PDCNet+ 生成稠密 `points3D.ply` 作为 CoR-GS + Binocular 的初始化点云。
