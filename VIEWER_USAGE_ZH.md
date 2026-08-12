# 渲染与自由视角查看

在项目根目录执行。默认使用 `gs0`，不需要填写 `--gaussian-index`。

## 一键启动自由视角

直接运行：

```powershell
python .\render.py
```

不带参数时自动打开自由视角查看器，默认读取根目录 `model\` 的最新模型和最新迭代。若模型在其他目录，使用 `--model-path` 指定。

## 渲染指定视角

```powershell
python .\render.py `
  --method corgs_fsgs `
  --dataset .\output\reconstruction\pipeline\dense `
  --model-path .\output\corgs\corgs_fsgs_midas_binocular_knn_11_4000 `
  --iteration 4000 `
  --views image_000001 `
  --render-depth
```

多个视角可以逗号分隔：

```powershell
python .\render.py --method corgs_fsgs --dataset .\output\reconstruction\pipeline\dense --model-path .\output\corgs\corgs_fsgs_midas_binocular_knn_11_4000 --iteration 4000 --views image_000001,image_000005,image_000009
```

结果默认保存到：

```text
<model-path>\train_gs0\ours_4000\renders\
```

## 自由视角查看

```powershell
python .\render.py `
  --interactive `
  --method corgs_fsgs `
  --dataset .\output\reconstruction\pipeline\dense `
  --model-path .\output\corgs\corgs_fsgs_midas_binocular_knn_11_4000 `
  --iteration 4000
```

- 鼠标左键拖动：旋转
- 鼠标右键拖动：平移
- 滚轮：缩放
- `R`：重置视角
- `S`：保存当前 RGB 图和位姿 JSON
- `Q` 或 `Esc`：退出

自由视角结果默认保存到：

```text
<model-path>\interactive\
```

`--gaussian-index 1` 仅用于旧的、已经保存 `point_cloud_gs2` 的双模型实验；新训练默认不保存 `gs1`，因此通常不要使用该参数。
