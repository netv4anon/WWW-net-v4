# NET-V4 配对预测评价接口

本接口将既有四臂数据和预测转换为可以独立使用的格式，不新增数据、训练或仿真。对象为256个构造上下文，每个上下文的B/P/R/F四臂各有12步历史与12步目标，共1024条输入。输入值均复用旧文件。

## 最短使用流程

Python 3.8及以上、NumPy 1.20及以上。进入本目录，在不存在同名输出文件/文件夹的条件下执行：

```powershell
python predict.py --inputs task/inputs.npz --adapter last_value.py:predict --out my_predictions.npz
python evaluate.py --task task --predictions my_predictions.npz --prediction-mode deterministic --out my_scores
```

将`last_value.py:predict`换成自己的Python文件与函数。函数接收`history`，形状为`(N,12)`，返回`(N,12)`有限实数预测。包装器只把历史数组传给函数，不传样本ID、臂、事件时间、分组或目标。权重载入由用户函数负责；不应利用评价数据重新训练或选参。

输入文件只有`sample_id`和`history`。ID为不含臂/上下文语义的传输标识，输入行顺序已打乱。另一种接入方式是自行写出含`sample_id`和`prediction`的NPZ；评分器按ID对齐，顺序不限，但必须一条不缺且一条不多。示例：

```python
np.savez_compressed('my_predictions.npz', sample_id=ids, prediction=forecasts)
```

`task/evaluation.npz`保存目标、ID与(context_id,arm)的映射、评分阶段及原始分割来源，只交给评分端。这里是接口数据分离，不是隔离任意恶意Python代码的安全沙箱，也不是新的盲测集。验证只证明示例程序在没有目标/元数据文件的临时目录中成功运行。

## 如何读结果

- `cross_losses.csv`：B/P/R/F历史产生的预测分别对四组目标计算MAE、RMSE和WMAPE，共4×4种比较。
- `decompositions.csv`：`D=L_FF-L_BB`，`T=L_BF-L_BB`，`A=L_BF-L_FF`，有`D=T-A`。`A>0`指改变历史后对同一F目标的误差降低，不是实测事故的因果效应。还报告反向路径与F/P差。
- `population_counts.csv`：全部、脉冲标记、脉冲前、脉冲后；按旧训练/验证/测试来源与发生时序分组。空格子显式记为`empty`，不记为零误差。阶段标签来自完整F构造，对所有交叉目标使用同一评分位置。
- `equal_history.json`：在B/F历史完全相同的95个上下文检查预测一致性。默认容差为`atol=1e-6,rtol=1e-7`；容差由使用者声明。差异只能触发数值/状态/额外输入检查，不能单凭差异断言泄漏。随机预测可声明`single_realization_stochastic`；单次随机输出不会得到确定性违规判决，需另行设计重复抽样比较。

指标在每个明确总体内按目标点微聚合。阶段均值或时序均值不能直接等权平均重建总分。WMAPE分母为零时留空。当前只支持12步点预测；不宣称支持概率预测或复现CiK的RCRPS。

全256上下文含162训练来源、46验证来源、48测试来源。这些是原开发暴露过的上下文，不是独立现场事件。成对样本有构造依赖；三个神经种子不等于三个独立算法。未来开始的事件导致相同历史，是该构造的已知属性。

## 已有结果与核验

`MODELS.json`列出11个不同保存配置：3个共享简单预测器，加两种训练快照各1个Ridge、3个STID种子。`saved_predictions`供独立评分；不把共享简单预测器重复计为两种方法。`results/primary_rows.csv`保留全上下文与测试来源结果，详细交叉评分位于模型子目录。

`VERIFICATION.json`核对旧脚本392行分解、6272行交叉损失、21952个数值，最大差为0。简单预测器被旧脚本按两种训练快照重复评分的行也逐项核对。新阶段切片单独检查了分区加和，不算旧脚本复现数量。

`export_existing.py`和`verify_interface.py`是开发阶段审计，需要原项目目录；独立包不靠这两个脚本运行。独立使用仅需要`evaluate.py`、`predict.py`、用户预测器、`task`及NumPy。保存预测重放入口是`replay_saved.py`。

这些检查证明软件复用正确，不证明科学原创性、事故因果识别或WWW投稿竞争力。相关工作与投稿判断见`NOVELTY_REVIEW_ZH.md`和`REPORT_ZH.md`。
