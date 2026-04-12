/// Main application state and egui UI logic.

use std::sync::{Arc, Mutex};

use egui::{Color32, RichText, TextEdit, Ui};
use egui_plot::{Legend, Plot};
use tokio::runtime::Runtime;

use crate::api_client::{BackendClient, IndicatorsResponse, OhlcvRecord, RealtimeQuote, StockInfo};
use crate::charts::{
    bb_lower_line, bb_upper_line, build_close_line, build_indicator_line, build_volume_bars,
    ema12_line, ema26_line, kdj_d_line, kdj_j_line, kdj_k_line, macd_hist_line, macd_line_chart,
    macd_signal_line, rsi14_line, sma10_line, sma20_line, sma5_line, COLOR_ALPHA,
};

#[derive(PartialEq, Clone, Copy)]
pub enum ActiveTab {
    Chart,
    Alpha,
    StockList,
}

pub struct StockApp {
    rt: Runtime,
    client: BackendClient,

    pub active_tab: ActiveTab,
    pub search_code: String,
    pub alpha_expr: String,
    pub status_msg: String,
    pub backend_alive: bool,

    pub quote: Arc<Mutex<Option<RealtimeQuote>>>,
    pub indicator_data: Arc<Mutex<Option<IndicatorsResponse>>>,
    pub alpha_dates: Arc<Mutex<Vec<String>>>,
    pub alpha_values: Arc<Mutex<Vec<Option<f64>>>>,
    pub stock_list: Arc<Mutex<Vec<StockInfo>>>,
    pub stock_filter: String,

    pub show_sma5: bool,
    pub show_sma10: bool,
    pub show_sma20: bool,
    pub show_ema12: bool,
    pub show_ema26: bool,
    pub show_bb: bool,
    pub show_macd: bool,
    pub show_rsi: bool,
    pub show_kdj: bool,
    pub show_volume: bool,
}

impl StockApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let rt = Runtime::new().expect("failed to create Tokio runtime");
        let client = BackendClient::default();

        let mut app = Self {
            rt,
            client,
            active_tab: ActiveTab::Chart,
            search_code: String::from("000001"),
            alpha_expr: String::from("rank(delta(close, 5)) / rank(volume)"),
            status_msg: String::from("Ready"),
            backend_alive: false,
            quote: Arc::new(Mutex::new(None)),
            indicator_data: Arc::new(Mutex::new(None)),
            alpha_dates: Arc::new(Mutex::new(Vec::new())),
            alpha_values: Arc::new(Mutex::new(Vec::new())),
            stock_list: Arc::new(Mutex::new(Vec::new())),
            stock_filter: String::new(),
            show_sma5: true,
            show_sma10: true,
            show_sma20: true,
            show_ema12: false,
            show_ema26: false,
            show_bb: false,
            show_macd: false,
            show_rsi: false,
            show_kdj: false,
            show_volume: true,
        };
        app.check_backend();
        app
    }

    fn check_backend(&mut self) {
        let client = self.client.clone();
        let alive = self.rt.block_on(async move { client.health().await });
        self.backend_alive = alive.unwrap_or(false);
        self.status_msg = if self.backend_alive {
            "Backend connected".to_owned()
        } else {
            "Backend not reachable – start with: python -m backend.main".to_owned()
        };
    }

    fn fetch_quote_and_indicators(&mut self) {
        let code = self.search_code.trim().to_owned();
        if code.is_empty() {
            return;
        }
        let client = self.client.clone();
        let quote_arc = self.quote.clone();
        let ind_arc = self.indicator_data.clone();

        let (quote_res, ind_res) = self.rt.block_on(async move {
            let q = client.quote(&code).await;
            let i = client.indicators(&code, None, None).await;
            (q, i)
        });

        match quote_res {
            Ok(q) => {
                self.status_msg = format!(
                    "{} ({}) 最新: {:.2}  涨跌: {:.2}%",
                    q.name,
                    q.code,
                    q.close,
                    if q.prev_close != 0.0 {
                        (q.close - q.prev_close) / q.prev_close * 100.0
                    } else {
                        0.0
                    }
                );
                *quote_arc.lock().unwrap() = Some(q);
            }
            Err(e) => self.status_msg = format!("Quote error: {e}"),
        }
        match ind_res {
            Ok(r) => *ind_arc.lock().unwrap() = Some(r),
            Err(e) => self.status_msg = format!("Indicator error: {e}"),
        }
    }

    fn fetch_alpha(&mut self) {
        let code = self.search_code.trim().to_owned();
        let expr = self.alpha_expr.trim().to_owned();
        if code.is_empty() || expr.is_empty() {
            return;
        }
        let client = self.client.clone();
        let dates_arc = self.alpha_dates.clone();
        let vals_arc = self.alpha_values.clone();

        let result = self.rt.block_on(async move { client.alpha(&code, &expr, None, None).await });
        match result {
            Ok(r) => {
                self.status_msg = format!("Alpha OK: {} points", r.values.len());
                *dates_arc.lock().unwrap() = r.dates;
                *vals_arc.lock().unwrap() = r.values;
            }
            Err(e) => self.status_msg = format!("Alpha error: {e}"),
        }
    }

    fn fetch_stock_list(&mut self) {
        let client = self.client.clone();
        let list_arc = self.stock_list.clone();
        let result = self.rt.block_on(async move { client.list_stocks().await });
        match result {
            Ok(list) => {
                self.status_msg = format!("Loaded {} stocks", list.len());
                *list_arc.lock().unwrap() = list;
            }
            Err(e) => self.status_msg = format!("Stock list error: {e}"),
        }
    }

    // ── UI panels ──────────────────────────────────────────────────────────

    fn ui_top_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if ui.selectable_label(self.active_tab == ActiveTab::Chart, "📈 K线图").clicked() {
                self.active_tab = ActiveTab::Chart;
            }
            if ui.selectable_label(self.active_tab == ActiveTab::Alpha, "⚡ Alpha因子").clicked() {
                self.active_tab = ActiveTab::Alpha;
            }
            if ui.selectable_label(self.active_tab == ActiveTab::StockList, "🔍 股票列表").clicked() {
                self.active_tab = ActiveTab::StockList;
                if self.stock_list.lock().unwrap().is_empty() {
                    self.fetch_stock_list();
                }
            }

            ui.separator();
            ui.label("股票代码:");
            let code_edit = ui.add(
                TextEdit::singleline(&mut self.search_code)
                    .desired_width(80.0)
                    .hint_text("000001"),
            );
            if (code_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                || ui.button("查询").clicked()
            {
                self.fetch_quote_and_indicators();
            }

            ui.separator();
            let color = if self.backend_alive { Color32::GREEN } else { Color32::RED };
            ui.label(RichText::new(&self.status_msg).color(color).small());

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("�� 检查后台").clicked() {
                    self.check_backend();
                }
            });
        });
    }

    fn ui_overlay_controls(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("均线:");
            ui.checkbox(&mut self.show_sma5, "MA5");
            ui.checkbox(&mut self.show_sma10, "MA10");
            ui.checkbox(&mut self.show_sma20, "MA20");
            ui.checkbox(&mut self.show_ema12, "EMA12");
            ui.checkbox(&mut self.show_ema26, "EMA26");
            ui.separator();
            ui.label("指标:");
            ui.checkbox(&mut self.show_bb, "布林线");
            ui.checkbox(&mut self.show_macd, "MACD");
            ui.checkbox(&mut self.show_rsi, "RSI");
            ui.checkbox(&mut self.show_kdj, "KDJ");
            ui.checkbox(&mut self.show_volume, "成交量");
        });
    }

    fn ui_quote_panel(&self, ui: &mut Ui) {
        if let Some(q) = self.quote.lock().unwrap().as_ref() {
            let pct = if q.prev_close != 0.0 {
                (q.close - q.prev_close) / q.prev_close * 100.0
            } else {
                0.0
            };
            let color = if pct >= 0.0 { Color32::RED } else { Color32::GREEN };
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} ({})", q.name, q.code)).strong());
                ui.label(RichText::new(format!("{:.2}", q.close)).color(color).heading());
                ui.label(RichText::new(format!("{:+.2}%", pct)).color(color));
                ui.separator();
                ui.label(format!("开:{:.2}", q.open));
                ui.label(format!("高:{:.2}", q.high));
                ui.label(format!("低:{:.2}", q.low));
                ui.label(format!("量:{:.0}手", q.volume / 100.0));
            });
        }
    }

    fn ui_chart_tab(&mut self, ui: &mut Ui) {
        self.ui_quote_panel(ui);
        ui.separator();
        self.ui_overlay_controls(ui);
        ui.separator();

        let ind_guard = self.indicator_data.lock().unwrap();
        if let Some(ind) = ind_guard.as_ref() {
            let records: Vec<OhlcvRecord> = ind
                .ohlcv
                .close
                .iter()
                .enumerate()
                .map(|(i, &c)| OhlcvRecord {
                    date: ind.dates.get(i).cloned().unwrap_or_default(),
                    open: *ind.ohlcv.open.get(i).unwrap_or(&c),
                    high: *ind.ohlcv.high.get(i).unwrap_or(&c),
                    low: *ind.ohlcv.low.get(i).unwrap_or(&c),
                    close: c,
                    volume: *ind.ohlcv.volume.get(i).unwrap_or(&0.0),
                    amount: None,
                    pct_chg: None,
                })
                .collect();

            let show_sma5 = self.show_sma5;
            let show_sma10 = self.show_sma10;
            let show_sma20 = self.show_sma20;
            let show_ema12 = self.show_ema12;
            let show_ema26 = self.show_ema26;
            let show_bb = self.show_bb;
            let available = ui.available_height();
            let price_h = available
                * if self.show_macd || self.show_rsi || self.show_kdj {
                    0.50
                } else {
                    0.70
                };

            Plot::new("price_chart")
                .height(price_h)
                .legend(Legend::default())
                .x_axis_label("日期")
                .y_axis_label("价格")
                .show(ui, |plot_ui| {
                    plot_ui.line(build_close_line(&records));
                    if show_sma5  { plot_ui.line(sma5_line(ind)); }
                    if show_sma10 { plot_ui.line(sma10_line(ind)); }
                    if show_sma20 { plot_ui.line(sma20_line(ind)); }
                    if show_ema12 { plot_ui.line(ema12_line(ind)); }
                    if show_ema26 { plot_ui.line(ema26_line(ind)); }
                    if show_bb {
                        plot_ui.line(bb_upper_line(ind));
                        plot_ui.line(bb_lower_line(ind));
                    }
                });

            if self.show_volume {
                Plot::new("volume_chart")
                    .height(available * 0.15)
                    .legend(Legend::default())
                    .y_axis_label("成交量")
                    .show(ui, |plot_ui| {
                        plot_ui.bar_chart(build_volume_bars(&records));
                    });
            }

            let sub_h = available * 0.12;

            if self.show_macd {
                Plot::new("macd_chart")
                    .height(sub_h)
                    .legend(Legend::default())
                    .y_axis_label("MACD")
                    .show(ui, |plot_ui| {
                        plot_ui.line(macd_line_chart(ind));
                        plot_ui.line(macd_signal_line(ind));
                        plot_ui.line(macd_hist_line(ind));
                    });
            }
            if self.show_rsi {
                Plot::new("rsi_chart")
                    .height(sub_h)
                    .legend(Legend::default())
                    .y_axis_label("RSI")
                    .show(ui, |plot_ui| {
                        plot_ui.line(rsi14_line(ind));
                    });
            }
            if self.show_kdj {
                Plot::new("kdj_chart")
                    .height(sub_h)
                    .legend(Legend::default())
                    .y_axis_label("KDJ")
                    .show(ui, |plot_ui| {
                        plot_ui.line(kdj_k_line(ind));
                        plot_ui.line(kdj_d_line(ind));
                        plot_ui.line(kdj_j_line(ind));
                    });
            }
        } else {
            ui.label("请输入股票代码并点击[查询]按钮");
        }
    }

    fn ui_alpha_tab(&mut self, ui: &mut Ui) {
        ui.heading("⚡ Alpha因子表达式");
        ui.label("支持: delay, delta, ts_mean, ts_std, ts_rank, ts_min, ts_max, ts_sum, rank, zscore, corr, abs, log, sign, sqrt, if_else");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("表达式:");
            ui.add(
                TextEdit::multiline(&mut self.alpha_expr)
                    .desired_width(500.0)
                    .desired_rows(2),
            );
            if ui.button("计算Alpha").clicked() {
                self.fetch_alpha();
            }
        });

        ui.separator();
        ui.label("示例（点击填入）:");
        let examples = [
            "rank(delta(close, 5)) / rank(volume)",
            "(close - delay(close, 20)) / delay(close, 20)",
            "ts_rank(volume, 20) - ts_rank(close, 20)",
            "corr(returns, log(volume), 10)",
            "zscore(delta(close, 1)) * sign(delta(volume, 1))",
        ];
        for ex in &examples {
            if ui.small_button(*ex).clicked() {
                self.alpha_expr = ex.to_string();
            }
        }

        ui.separator();

        let vals_guard = self.alpha_values.lock().unwrap();
        if !vals_guard.is_empty() {
            let title = format!("Alpha: {}", self.alpha_expr);
            let vals: Vec<Option<f64>> = vals_guard.clone();
            drop(vals_guard);
            Plot::new("alpha_chart")
                .height(ui.available_height() - 20.0)
                .legend(Legend::default())
                .y_axis_label("Alpha")
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        build_indicator_line(&vals, &title, COLOR_ALPHA).width(2.0),
                    );
                });
        } else {
            ui.label("请先查询股票，然后输入Alpha表达式并点击[计算Alpha]");
        }
    }

    fn ui_stock_list_tab(&mut self, ui: &mut Ui) {
        ui.heading("🔍 A股股票列表");

        ui.horizontal(|ui| {
            ui.label("搜索:");
            ui.text_edit_singleline(&mut self.stock_filter);
            if ui.button("刷新列表").clicked() {
                self.fetch_stock_list();
            }
        });

        let filter = self.stock_filter.to_lowercase();
        let list_guard = self.stock_list.lock().unwrap();

        if list_guard.is_empty() {
            ui.label("列表加载中...");
            return;
        }

        let filtered: Vec<(String, String, String)> = list_guard
            .iter()
            .filter(|s| {
                filter.is_empty()
                    || s.code.contains(&filter)
                    || s.name.to_lowercase().contains(&filter)
            })
            .take(200)
            .map(|s| (s.code.clone(), s.name.clone(), s.exchange.clone()))
            .collect();

        let total = list_guard.len();
        drop(list_guard);

        ui.label(format!("显示 {}/{} 只股票", filtered.len(), total));
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("stock_grid")
                .striped(true)
                .num_columns(3)
                .show(ui, |ui| {
                    ui.strong("代码");
                    ui.strong("名称");
                    ui.strong("交易所");
                    ui.end_row();

                    let mut clicked_code: Option<String> = None;
                    for (code, name, exch) in &filtered {
                        if ui.link(RichText::new(code).monospace()).on_hover_text("点击查看K线").clicked() {
                            clicked_code = Some(code.clone());
                        }
                        ui.label(name);
                        ui.label(exch);
                        ui.end_row();
                    }
                    if let Some(c) = clicked_code {
                        self.search_code = c;
                        self.active_tab = ActiveTab::Chart;
                        self.fetch_quote_and_indicators();
                    }
                });
        });
    }
}

impl eframe::App for StockApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            self.ui_top_bar(ui);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.active_tab {
                ActiveTab::Chart => self.ui_chart_tab(ui),
                ActiveTab::Alpha => self.ui_alpha_tab(ui),
                ActiveTab::StockList => self.ui_stock_list_tab(ui),
            }
        });
    }
}
