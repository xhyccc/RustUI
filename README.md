# RustUI – A股量价分析工具

A stock analysis tool combining a **Python FastAPI backend** (data, indicators, alpha) with a **Rust egui desktop UI** (charts, interaction).

## Features

| Feature | Details |
|---------|---------|
| **数据源** | yfinance · akshare · 新浪财经实时行情 |
| **本地缓存** | SQLite（TTL可配置，避免重复请求） |
| **K线图** | 收盘价折线 + 日成交量柱 |
| **均线 / 指标** | MA5/10/20/60/120/250 · EMA · MACD · RSI · 布林线 · KDJ · ATR · OBV · VWAP · CCI |
| **Alpha因子** | 自定义表达式引擎，支持 delay/delta/ts_rank/corr/rank/zscore 等 |
| **安全沙箱** | Alpha表达式 AST 白名单校验，禁止任意代码执行 |

---

## 快速开始

### 1. 安装 Python 依赖
```bash
pip install -r backend/requirements.txt
```

### 2. 启动 Python 后台
```bash
python -m backend.main
# 默认监听 http://127.0.0.1:8000
```

### 3. 运行 Rust UI
```bash
cargo run --release
```

---

## API 文档（后台）

启动后台后访问 http://127.0.0.1:8000/docs 查看交互式 Swagger 文档。

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/stocks` | A股股票列表 |
| GET | `/api/quote/{code}` | 实时行情（新浪） |
| POST | `/api/quotes/batch` | 批量实时行情 |
| GET | `/api/history/{code}` | 日线历史OHLCV |
| GET | `/api/intraday/{code}` | 分钟线 |
| GET | `/api/yfinance/{ticker}` | yfinance 历史 |
| GET | `/api/indicators/{code}` | 所有技术指标 |
| POST | `/api/alpha` | 自定义Alpha因子 |

---

## Alpha因子表达式示例

```
rank(delta(close, 5)) / rank(volume)
(close - delay(close, 20)) / delay(close, 20)
ts_rank(volume, 20) - ts_rank(close, 20)
corr(returns, log(volume), 10)
zscore(delta(close, 1)) * sign(delta(volume, 1))
```

Supported functions: delay, delta, ts_mean, ts_std, ts_rank, ts_min, ts_max, ts_sum, rank, zscore, corr, abs, log, sign, sqrt, pow, if_else

---

## 测试

```bash
# Python 测试 (69 tests)
python -m pytest backend/tests/ -v

# Rust 测试 (4 integration tests)
cargo test
```

---

## 项目结构

```
RustUI/
├── Cargo.toml
├── src/
│   ├── main.rs          # Rust 入口
│   ├── lib.rs           # 公共库（供测试）
│   ├── app.rs           # egui 应用主界面
│   ├── api_client.rs    # HTTP 客户端
│   └── charts.rs        # 图表渲染
├── tests/
│   └── integration_test.rs
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI 服务器
│   ├── data_sources.py  # 数据获取（yfinance/akshare/新浪）
│   ├── cache.py         # SQLite 缓存
│   ├── indicators.py    # 技术指标
│   ├── alpha_engine.py  # Alpha 因子引擎
│   ├── requirements.txt
│   └── tests/
│       ├── test_cache.py
│       ├── test_indicators.py
│       ├── test_alpha_engine.py
│       └── test_api.py
└── pyproject.toml
```
