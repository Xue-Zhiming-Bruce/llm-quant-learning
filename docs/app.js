const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};

const formatPct = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const className = Number(value) >= 0 ? "positive" : "negative";
  const sign = Number(value) > 0 ? "+" : "";
  return `<span class="${className}">${sign}${formatNumber(value)}%</span>`;
};

const setText = (id, value) => {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
};

const setHTML = (id, value) => {
  const element = document.getElementById(id);
  if (element) element.innerHTML = value;
};

const fillList = (id, items) => {
  const element = document.getElementById(id);
  if (!element) return;
  element.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
};

const fillKeyValues = (id, rows) => {
  const element = document.getElementById(id);
  if (!element) return;
  element.innerHTML = rows.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
};

const initTabs = () => {
  const buttons = [...document.querySelectorAll(".tab-button")];
  const panels = [...document.querySelectorAll(".panel")];

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      buttons.forEach((item) => item.classList.toggle("active", item === button));
      panels.forEach((panel) => panel.classList.toggle("active", panel.id === tab));
    });
  });
};

const renderTable = (rows) => {
  const body = document.getElementById("recent-table");
  if (!body) return;

  body.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.trade_date}</td>
          <td>${formatNumber(row.open)}</td>
          <td>${formatNumber(row.high)}</td>
          <td>${formatNumber(row.low)}</td>
          <td>${formatNumber(row.close)}</td>
          <td>${formatPct(row.pct_chg)}</td>
          <td>${formatNumber(row.vol)}</td>
        </tr>
      `,
    )
    .join("");
};

const renderDashboard = (data) => {
  setText("disclaimer", data.disclaimer);
  setText("latest-date", data.price.latest_date);
  setText("latest-close", formatNumber(data.price.latest_close));
  setHTML("latest-change", formatPct(data.price.latest_change_pct));
  setHTML("one-year-return", formatPct(data.price.one_year_return_pct));
  setHTML("return-20d", formatPct(data.price.return_20d_pct));
  setHTML("return-60d", formatPct(data.price.return_60d_pct));
  setHTML("max-drawdown", formatPct(data.price.max_drawdown_pct));
  setText("trend-label", data.technical.trend_label);
  setText("market-cap", data.fundamental.latest_daily_basic.total_mv_yi ? `${formatNumber(data.fundamental.latest_daily_basic.total_mv_yi)} 亿元` : "--");
  setText("generated-at", `生成时间：${data.generated_at}`);

  fillList("overview-summary", [
    ...data.technical.summary.slice(0, 2),
    ...data.fundamental.summary.slice(0, 2),
  ]);
  fillList("technical-summary", data.technical.summary);
  fillList("kline-summary", data.kline.summary);
  fillList("fundamental-summary", data.fundamental.summary);

  fillKeyValues("technical-values", [
    ["MA5", formatNumber(data.technical.ma5)],
    ["MA20", formatNumber(data.technical.ma20)],
    ["MA60", formatNumber(data.technical.ma60)],
    ["MA120", formatNumber(data.technical.ma120)],
    ["RSI14", `${formatNumber(data.technical.rsi14)} · ${data.technical.rsi_label}`],
    ["MACD", `${formatNumber(data.technical.macd_diff, 3)} / ${formatNumber(data.technical.macd_dea, 3)} / ${formatNumber(data.technical.macd_hist, 3)}`],
    ["20 日年化波动", formatPct(data.technical.volatility20_pct)],
    ["量能/20日均量", `${formatNumber(data.technical.volume_vs_ma20)} 倍`],
  ]);

  fillKeyValues("company-values", [
    ["证券代码", data.stock.ts_code],
    ["公司简称", data.stock.name],
    ["地区", data.stock.area],
    ["行业", data.stock.industry],
    ["板块", data.stock.market],
    ["上市日期", data.stock.list_date],
    ["业务备注", data.stock.business_note],
  ]);

  const valuation = data.fundamental.latest_daily_basic;
  fillKeyValues("valuation-values", [
    ["估值日期", valuation.trade_date || "--"],
    ["PE(TTM)", formatNumber(valuation.pe_ttm)],
    ["PE", formatNumber(valuation.pe)],
    ["PB", formatNumber(valuation.pb)],
    ["PS(TTM)", formatNumber(valuation.ps_ttm)],
    ["换手率", formatPct(valuation.turnover_rate)],
    ["量比", formatNumber(valuation.volume_ratio)],
    ["总市值", valuation.total_mv_yi ? `${formatNumber(valuation.total_mv_yi)} 亿元` : "--"],
  ]);

  setText("data-status", data.fundamental.data_status);
  const sourceList = document.getElementById("source-list");
  if (sourceList) {
    sourceList.innerHTML = data.fundamental.sources
      .map((source) => `<li><a href="${source.url}" target="_blank" rel="noopener noreferrer">${source.name}</a></li>`)
      .join("");
  }

  renderTable(data.recent_rows);
};

initTabs();

fetch("data/analysis.json")
  .then((response) => response.json())
  .then(renderDashboard)
  .catch(() => {
    setText("generated-at", "数据加载失败");
  });
