# TASK6：机器学习量化交易策略

本目录使用 `model_data.csv` 中的股票季度基本面因子预测下一季度收益率，比较岭回归、决策树和随机森林，并在严格的时间外测试集上回测选股策略。

## 运行

在项目根目录执行：

```bash
python TASK6/analyze_ml_strategy.py
```

默认设置：

- 前 70% 的完整季度用于训练，后 30% 用于测试；
- 每个季度买入预测收益最高的 20% 股票，等权持有一个季度；
- 按换手率扣除单边 0.1% 的交易成本；
- 输出位于 `TASK6/output/`，图形位于 `TASK6/figures/`。

可调整参数：

```bash
python TASK6/analyze_ml_strategy.py \
  --train-fraction 0.7 \
  --selection-fraction 0.2 \
  --transaction-cost 0.001
```

完整理论说明、变量定义、方法、结果和局限性见 `analysis_report.md`。
