# CoR-GS + FSGS + Binocular 联合实验

## 实验配置

- 输入：当前 `data` 中的 11 张照片；COLMAP 最终注册 9 张。
- 模式：`corgs_fsgs`。
- CoR-GS：两个 Gaussian 分支 `gs0` / `gs1`，启用伪视角互相对比和共同剪枝。
- FSGS：启用 `fsgs_unpool_n=3`，前 2000 次训练进行增密；启用 MiDaS v2.1 Small 单目深度约束。
- Binocular：第 2500 次开始，每 20 次生成虚拟右相机并加入双目一致性损失，权重 `0.2`。
- 伪视角：普通 COLMAP `dense` 数据生成 255 个可用伪相机；CoR-GS 伪视角对比和 FSGS 伪视角深度分支均实际运行。

## 结果

- 输出目录：`output/corgs/corgs_fsgs_midas_binocular_knn_11_4000`。
- 训练迭代：4000。
- 最终训练损失：约 `0.0929`。
- `gs0` Gaussian 数量：234,575。
- `gs1` Gaussian 数量：233,556。
- `gs0` 点云：`point_cloud/iteration_4000/point_cloud.ply`。
- `gs1` 点云：`point_cloud_gs2/iteration_4000/point_cloud.ply`。
- 双模型渲染：`train_gs0/ours_4000/renders/image_000001.png`、`train_gs1/ours_4000/renders/image_000001.png`。
- 9 个训练视角平均 PSNR：`gs0=27.311 dB`，`gs1=27.407 dB`；这是训练视角指标，不是独立新视角评估。

当前没有留出测试相机，因此最终损失只能作为训练稳定性指标，不能单独证明新视角质量改善。

后续训练默认只保存 `gs0` 的最终点云；`gs1` 仍参与联合训练，但不会写入 `point_cloud_gs2`。暂停检查点仍保留双分支状态，以支持联合训练断点恢复。
