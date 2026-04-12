/// RustUI – A-share stock analysis desktop application.
///
/// Architecture:
///   - Python FastAPI backend (backend/) provides REST API for data and indicators
///   - This Rust binary provides the native desktop UI via egui
///
/// Usage:
///   1. Start the Python backend:   python -m backend.main
///   2. Run this binary:             cargo run --release

mod api_client;
mod app;
mod charts;

use tracing::info;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting RustUI");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("RustUI – A股量价分析")
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "RustUI",
        options,
        Box::new(|cc| Ok(Box::new(app::StockApp::new(cc)))),
    )
}
