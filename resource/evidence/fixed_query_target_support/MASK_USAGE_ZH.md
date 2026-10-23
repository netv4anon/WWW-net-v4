# 固定测试集的训练行号用法

TRAIN_ROWS.npz 的六个键分别是 Original、WholeTraffic、TargetSupport、Random4101、Random4102、Random4103。值为零起始的原始训练行号，均位于 [0,3909)；不是重排后的索引。固定测试为原文件 [4746,5585)，839 行，验证不参与本轮参数选择。

MASK_CONTRACT.json 记录原文件字节数、SHA256、全部行号及语义。先验证源文件完全一致，再使用这些行号；不能用于另一地区、版本或顺序。

    import numpy as np
    rows = np.load("TRAIN_ROWS.npz", allow_pickle=False)["TargetSupport"]
    assert len(rows) == 2665

TEST_GROUPS.npz 则保存测试切片内的局部行号，不应直接当原文件全局行号。分组为全部、完整交通复用、其他目标复用和无原训练目标复用，共 93/466/280 三个互斥组。

行号由测试目标构造，是暴露开发数据上的离线实验。它排除了声明的全路网状态字节重复，不恢复日期、不保证未来预测独立性，也不是可部署的在线选样策略。

prepare_support.py 和 verify_support.py 从固定原文件验证相等支持及全部保留集；fit_probe.py、score_probe.py、verify_probe.py 实现固定 AR-Ridge 及独立 QR 核验。脚本依赖本项目中摘要绑定的来源路径，尚未声称可以脱离项目独立运行。重复执行时需复制到新的阶段目录，不能覆盖 START、FIT_START、预测封存或既有结果。用户可在原文件载入后用所附行号建立自己的训练条件，不应据本轮测试选择更有利的模型或掩码。
