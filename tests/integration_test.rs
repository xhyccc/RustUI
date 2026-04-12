/// Integration tests for the Rust api_client module.
///
/// These tests use a mock HTTP server to verify that the BackendClient
/// correctly serialises requests and deserialises responses.

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use serde_json::json;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Spin up a minimal one-shot HTTP server on a random port.
    /// Returns (port, join-handle).
    async fn mock_server(response_body: serde_json::Value) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let body = response_body.to_string();

        let handle = tokio::task::spawn_blocking(move || {
            let (mut stream, _) = listener.accept().unwrap();
            use std::io::{Read, Write};
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(resp.as_bytes()).unwrap();
        });

        (port, handle)
    }

    // ── health ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_health_ok() {
        let (port, handle) = mock_server(json!({"status": "ok", "cache": {}})).await;
        let client = rustui::BackendClient::new(format!("http://127.0.0.1:{}", port));
        let result = client.health().await;
        let _ = handle.await;
        assert!(result.unwrap());
    }

    // ── quote ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_quote_deserialization() {
        let (port, handle) = mock_server(json!({
            "code": "000001",
            "name": "平安银行",
            "open": 12.0,
            "prev_close": 11.9,
            "close": 12.5,
            "high": 12.6,
            "low": 11.8,
            "bid": 12.49,
            "ask": 12.51,
            "volume": 50000000.0,
            "amount": 625000000.0,
            "date": "2024-01-15",
            "time": "15:00:00"
        }))
        .await;
        let client = rustui::BackendClient::new(format!("http://127.0.0.1:{}", port));
        let q = client.quote("000001").await.unwrap();
        let _ = handle.await;
        assert_eq!(q.code, "000001");
        assert_eq!(q.name, "平安银行");
        assert!((q.close - 12.5).abs() < 1e-9);
    }

    // ── history ───────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_history_deserialization() {
        let (port, handle) = mock_server(json!({
            "code": "000001",
            "data": [
                {"date": "2024-01-02", "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 1000000.0},
                {"date": "2024-01-03", "open": 10.5, "high": 11.5, "low": 10.0, "close": 11.0, "volume": 1200000.0}
            ]
        }))
        .await;
        let client = rustui::BackendClient::new(format!("http://127.0.0.1:{}", port));
        let h = client.history("000001", None, None, "qfq").await.unwrap();
        let _ = handle.await;
        assert_eq!(h.code, "000001");
        assert_eq!(h.data.len(), 2);
        assert_eq!(h.data[0].date, "2024-01-02");
    }

    // ── alpha ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_alpha_deserialization() {
        let (port, handle) = mock_server(json!({
            "code": "000001",
            "expression": "rank(close)",
            "dates": ["2024-01-02", "2024-01-03"],
            "values": [0.3, 0.7]
        }))
        .await;
        let client = rustui::BackendClient::new(format!("http://127.0.0.1:{}", port));
        let a = client
            .alpha("000001", "rank(close)", None, None)
            .await
            .unwrap();
        let _ = handle.await;
        assert_eq!(a.expression, "rank(close)");
        assert_eq!(a.values.len(), 2);
        assert!((a.values[1].unwrap() - 0.7).abs() < 1e-9);
    }
}
