/// HTTP client for the Python FastAPI backend.
///
/// All methods are async and return `anyhow::Result`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Base URL of the backend server.
pub const BACKEND_URL: &str = "http://127.0.0.1:8000";

// ─────────────────────────────────────────────────────────────────────────────
// Response types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone, Default)]
pub struct RealtimeQuote {
    pub code: String,
    pub name: String,
    pub open: f64,
    pub prev_close: f64,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub bid: f64,
    pub ask: f64,
    pub volume: f64,
    pub amount: f64,
    pub date: String,
    pub time: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OhlcvRecord {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub amount: Option<f64>,
    pub pct_chg: Option<f64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct HistoryResponse {
    pub code: String,
    pub data: Vec<OhlcvRecord>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OhlcvSeries {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IndicatorsResponse {
    pub code: String,
    pub dates: Vec<String>,
    pub ohlcv: OhlcvSeries,
    pub indicators: HashMap<String, Vec<Option<f64>>>,
}

#[derive(Debug, Serialize)]
pub struct AlphaRequest {
    pub code: String,
    pub expression: String,
    pub start: Option<String>,
    pub end: Option<String>,
    pub adjust: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AlphaResponse {
    pub code: String,
    pub expression: String,
    pub dates: Vec<String>,
    pub values: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StockInfo {
    pub code: String,
    pub name: String,
    pub exchange: String,
}

#[derive(Debug, Deserialize)]
pub struct StockListResponse {
    pub data: Vec<StockInfo>,
}

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Client
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct BackendClient {
    client: reqwest::Client,
    base: String,
}

impl BackendClient {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .expect("failed to build reqwest client"),
            base: base.into(),
        }
    }

    pub fn default() -> Self {
        Self::new(BACKEND_URL)
    }

    /// GET /health
    pub async fn health(&self) -> anyhow::Result<bool> {
        let resp: HealthResponse = self
            .client
            .get(format!("{}/health", self.base))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp.status == "ok")
    }

    /// GET /api/stocks
    pub async fn list_stocks(&self) -> anyhow::Result<Vec<StockInfo>> {
        let resp: StockListResponse = self
            .client
            .get(format!("{}/api/stocks", self.base))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp.data)
    }

    /// GET /api/quote/{code}
    pub async fn quote(&self, code: &str) -> anyhow::Result<RealtimeQuote> {
        let q: RealtimeQuote = self
            .client
            .get(format!("{}/api/quote/{}", self.base, code))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(q)
    }

    /// GET /api/history/{code}
    pub async fn history(
        &self,
        code: &str,
        start: Option<&str>,
        end: Option<&str>,
        adjust: &str,
    ) -> anyhow::Result<HistoryResponse> {
        let mut url = format!("{}/api/history/{}?adjust={}", self.base, code, adjust);
        if let Some(s) = start {
            url.push_str(&format!("&start={}", s));
        }
        if let Some(e) = end {
            url.push_str(&format!("&end={}", e));
        }
        let r: HistoryResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(r)
    }

    /// GET /api/indicators/{code}
    pub async fn indicators(
        &self,
        code: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> anyhow::Result<IndicatorsResponse> {
        let mut url = format!("{}/api/indicators/{}", self.base, code);
        let mut sep = '?';
        if let Some(s) = start {
            url.push_str(&format!("{}start={}", sep, s));
            sep = '&';
        }
        if let Some(e) = end {
            url.push_str(&format!("{}end={}", sep, e));
        }
        let r: IndicatorsResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(r)
    }

    /// POST /api/alpha
    pub async fn alpha(
        &self,
        code: &str,
        expression: &str,
        start: Option<String>,
        end: Option<String>,
    ) -> anyhow::Result<AlphaResponse> {
        let body = AlphaRequest {
            code: code.to_owned(),
            expression: expression.to_owned(),
            start,
            end,
            adjust: "qfq".to_owned(),
        };
        let r: AlphaResponse = self
            .client
            .post(format!("{}/api/alpha", self.base))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(r)
    }
}
