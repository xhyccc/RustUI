/// Chart rendering helpers for egui_plot.

use egui::Color32;
use egui_plot::{Bar, BarChart, Line, PlotPoints};

use crate::api_client::{IndicatorsResponse, OhlcvRecord};

pub const COLOR_UP: Color32 = Color32::from_rgb(220, 50, 50);
pub const COLOR_DOWN: Color32 = Color32::from_rgb(0, 180, 100);
pub const COLOR_SMA5: Color32 = Color32::from_rgb(255, 215, 0);
pub const COLOR_SMA10: Color32 = Color32::from_rgb(255, 165, 0);
pub const COLOR_SMA20: Color32 = Color32::from_rgb(100, 180, 255);
pub const COLOR_EMA12: Color32 = Color32::from_rgb(200, 100, 255);
pub const COLOR_EMA26: Color32 = Color32::from_rgb(255, 100, 200);
pub const COLOR_ALPHA: Color32 = Color32::from_rgb(255, 255, 100);

pub fn build_candlestick_bars(records: &[OhlcvRecord]) -> BarChart {
    let bars: Vec<Bar> = records
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let color = if r.close >= r.open { COLOR_UP } else { COLOR_DOWN };
            Bar::new(i as f64, r.close)
                .width(0.8)
                .fill(color)
                .stroke(egui::Stroke::new(1.0, color))
        })
        .collect();
    BarChart::new(bars).name("K线")
}

pub fn build_indicator_line<'a>(values: &'a [Option<f64>], name: &'a str, color: Color32) -> Line<'a> {
    let points: PlotPoints = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|y| [i as f64, y]))
        .collect::<Vec<_>>()
        .into();
    Line::new(points).name(name).color(color).width(1.5)
}

pub fn build_close_line(records: &[OhlcvRecord]) -> Line<'static> {
    let points: PlotPoints = records
        .iter()
        .enumerate()
        .map(|(i, r)| [i as f64, r.close])
        .collect::<Vec<_>>()
        .into();
    Line::new(points).name("收盘价").color(Color32::WHITE).width(1.0)
}

pub fn build_volume_bars(records: &[OhlcvRecord]) -> BarChart {
    let bars: Vec<Bar> = records
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let color = if r.close >= r.open { COLOR_UP } else { COLOR_DOWN };
            Bar::new(i as f64, r.volume)
                .width(0.8)
                .fill(color)
                .stroke(egui::Stroke::new(0.5, color))
        })
        .collect();
    BarChart::new(bars).name("成交量")
}

fn get_series(r: &IndicatorsResponse, name: &str) -> Vec<Option<f64>> {
    r.indicators.get(name).cloned().unwrap_or_default()
}

fn owned_line(values: Vec<Option<f64>>, name: &'static str, color: Color32) -> Line<'static> {
    let points: PlotPoints = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|y| [i as f64, y]))
        .collect::<Vec<_>>()
        .into();
    Line::new(points).name(name).color(color).width(1.5)
}

pub fn sma5_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "sma5"), "MA5", COLOR_SMA5) }
pub fn sma10_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "sma10"), "MA10", COLOR_SMA10) }
pub fn sma20_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "sma20"), "MA20", COLOR_SMA20) }
pub fn ema12_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "ema12"), "EMA12", COLOR_EMA12) }
pub fn ema26_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "ema26"), "EMA26", COLOR_EMA26) }
pub fn bb_upper_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "bb_upper"), "BB上轨", Color32::from_rgb(150, 150, 255)) }
pub fn bb_lower_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "bb_lower"), "BB下轨", Color32::from_rgb(150, 150, 255)) }
pub fn macd_line_chart(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "macd"), "MACD", Color32::from_rgb(255, 200, 0)) }
pub fn macd_signal_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "macd_signal"), "Signal", Color32::from_rgb(200, 100, 255)) }
pub fn macd_hist_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "macd_hist"), "柱", Color32::from_rgb(150, 220, 150)) }
pub fn rsi14_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "rsi14"), "RSI14", Color32::from_rgb(100, 220, 220)) }
pub fn kdj_k_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "kdj_k"), "K", Color32::from_rgb(255, 200, 0)) }
pub fn kdj_d_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "kdj_d"), "D", Color32::from_rgb(255, 100, 100)) }
pub fn kdj_j_line(r: &IndicatorsResponse) -> Line<'static> { owned_line(get_series(r, "kdj_j"), "J", Color32::from_rgb(100, 255, 100)) }
