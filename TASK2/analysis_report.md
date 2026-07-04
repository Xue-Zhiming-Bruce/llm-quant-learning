# TASK2：基础诊断与技术指标分析

## 1. 数据基础诊断

本次分析读取 `TASK2` 目录下已存储的两份日线行情数据，字段包括 `open`、`high`、`low`、`close`、`pre_close`、`pct_chg`、`vol`、`amount` 等。数据按 `trade_date` 升序排列后再计算指标。

| 源文件 | 股票名称 | 代码 | 行数 | 开始日期 | 结束日期 | 重复交易日 | 缺失值总数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 三一重工行情数据.csv | 三一重工 | 600031.SH | 129 | 2025-01-02 | 2025-07-16 | 0 | 0 |
| 平安集团行情数据.csv | 平安集团 | 000001.SZ | 372 | 2024-01-02 | 2025-07-17 | 0 | 0 |

更完整的诊断结果已保存：

- `output/diagnostics_summary.csv`
- `output/missing_values.csv`
- `output/descriptive_statistics.csv`

## 2. RSI、MACD、布林带指标说明

### RSI：相对强弱指标

RSI 是动量震荡指标，衡量最近一段时间上涨力度与下跌力度的相对强弱，常用周期为 14。计算步骤：

```text
Gain_t = max(Close_t - Close_(t-1), 0)
Loss_t = max(Close_(t-1) - Close_t, 0)

AvgGain_t = (AvgGain_(t-1) * 13 + Gain_t) / 14
AvgLoss_t = (AvgLoss_(t-1) * 13 + Loss_t) / 14

RS = AvgGain / AvgLoss
RSI = 100 - 100 / (1 + RS)
```

作用：RSI 通常用于观察超买超卖、短期动能变化和背离。常见参考线是 70 和 30，但强趋势中 RSI 可能长时间处于高位或低位，因此不能单独作为买卖依据。

### MACD：指数平滑异同移动平均线

MACD 用短周期 EMA 与长周期 EMA 的差值观察趋势和动能变化，常用参数为 12、26、9：

```text
EMA_t = alpha * Close_t + (1 - alpha) * EMA_(t-1)
alpha = 2 / (N + 1)

MACD Line = EMA_12 - EMA_26
Signal Line = EMA_9(MACD Line)
Histogram = MACD Line - Signal Line
```

作用：MACD 常用于判断趋势方向、金叉死叉、动能增强或衰减。震荡行情中容易出现反复假信号。

### 布林带：Bollinger Bands

布林带用均线和标准差描述价格的相对高低和波动区间，常用参数为 20 日均线和 2 倍标准差：

```text
Middle Band = SMA_20
Upper Band = SMA_20 + 2 * Std_20
Lower Band = SMA_20 - 2 * Std_20

Bandwidth = (Upper Band - Lower Band) / Middle Band
%B = (Close - Lower Band) / (Upper Band - Lower Band)
```

作用：布林带可以观察价格相对位置、波动率收缩和扩张，也可辅助识别突破或均值回归机会。价格触及上下轨不等于一定反转。

## 3. Python 实现和可视化输出

脚本位置：`analyze_technical_indicators.py`

脚本完成了：

1. 加载已存储的股价 CSV；
2. 检查缺失值、重复交易日，并计算描述性统计量；
3. 计算 RSI(14)、MACD(12,26,9)、布林带(20,2)；
4. 扩展计算 ATR(14)；
5. 输出指标明细 CSV 和技术指标图。

生成图形：

- `figures/600031.SH_technical_indicators.png`
- `figures/000001.SZ_technical_indicators.png`

最新交易日指标摘要：

| 股票名称 | 代码 | 日期 | 收盘价 | RSI14 | MACD | Signal | Hist | BB上轨 | BB中轨 | BB下轨 | ATR14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 三一重工 | 600031.SH | 2025-07-16 | 18.7100 | 55.6464 | 0.1559 | 0.0609 | 0.0951 | 19.3031 | 18.2670 | 17.2309 | 0.3253 |
| 平安集团 | 000001.SZ | 2025-07-17 | 12.5900 | 56.3944 | 0.2715 | 0.2900 | -0.0185 | 13.2181 | 12.4675 | 11.7169 | 0.2607 |

## 4. 其他典型指标与扩展指标 ATR

量化中常见指标还包括：

- 均线 MA / EMA：判断趋势方向和均线交叉。
- 动量 Momentum / ROC：衡量当前价格相对过去价格的涨跌幅。
- KDJ / Stochastic Oscillator：观察收盘价在最近高低区间中的位置。
- ADX：判断趋势强弱，不直接判断方向。
- OBV：结合成交量观察资金流入流出趋势。
- VWAP：成交量加权平均价，常用于日内交易和执行交易评价。
- 波动率 Volatility：用收益率标准差衡量风险。
- MFI：结合价格和成交量的资金流量指标。

本次扩展选取 ATR（Average True Range，平均真实波幅）。ATR 由 Welles Wilder 提出，衡量价格真实波动幅度，不判断涨跌方向。计算方法：

```text
TR_t = max(
  High_t - Low_t,
  abs(High_t - Close_(t-1)),
  abs(Low_t - Close_(t-1))
)

ATR_t = (ATR_(t-1) * 13 + TR_t) / 14
```

作用：ATR 常用于动态止损、仓位控制、识别波动扩大或收缩。例如趋势策略中可以用 `2 * ATR` 作为止损距离，避免固定金额止损无法适应市场波动。

## 参考资料

- Investopedia, Relative Strength Index (RSI): https://www.investopedia.com/terms/r/rsi.asp
- Investopedia, Moving Average Convergence Divergence (MACD): https://www.investopedia.com/terms/m/macd.asp
- Wikipedia, Bollinger Bands: https://en.wikipedia.org/wiki/Bollinger_Bands
- Wikipedia, Average True Range: https://en.wikipedia.org/wiki/Average_true_range
