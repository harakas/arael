// MCP (Model Context Protocol) server embedded in the sketch editor.
// Runs an async HTTP server (axum/tokio) in a background thread.
// Communicates with the GUI thread via channels.
//
// Protocol: Streamable HTTP (MCP spec 2025-03-26)
// Spec: https://modelcontextprotocol.io/specification/

#![cfg(not(target_arch = "wasm32"))]

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode, Method, Uri},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post, delete},
    Json, Router,
};
use eframe::egui;
use serde_json::{json, Value};

/// A request from the MCP server to the GUI thread.
pub struct McpRequest {
    pub command: String,
    pub response_tx: oneshot::Sender<String>,
    pub blocked_commands: Vec<&'static str>,
}

#[derive(Clone)]
struct McpState {
    tx: Arc<mpsc::Sender<McpRequest>>,
    verbose: bool,
    egui_ctx: Arc<Mutex<Option<egui::Context>>>,
    session_id: Arc<Mutex<Option<String>>>,
}

/// Start the MCP server on the given address.
/// Returns a receiver for commands that the GUI thread should poll.
pub fn start(
    addr: SocketAddr,
    verbose: bool,
    egui_ctx: Arc<Mutex<Option<egui::Context>>>,
) -> mpsc::Receiver<McpRequest> {
    let (tx, rx) = mpsc::channel::<McpRequest>(32);
    let state = McpState {
        tx: Arc::new(tx),
        verbose,
        egui_ctx,
        session_id: Arc::new(Mutex::new(None)),
    };
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new()
            .expect("Failed to create tokio runtime for MCP server");
        rt.block_on(async move {
            let app = Router::new()
                .route("/mcp", post(handle_post))
                .route("/mcp", get(handle_get))
                .route("/mcp", delete(handle_delete))
                .fallback(handle_fallback)
                .layer(middleware::from_fn_with_state(state.clone(), log_middleware))
                .with_state(state);
            let listener = tokio::net::TcpListener::bind(addr).await
                .unwrap_or_else(|e| panic!("MCP server failed to bind {}: {}", addr, e));
            eprintln!("MCP server listening on http://{}/mcp", addr);
            axum::serve(listener, app).await.unwrap();
        });
    });
    rx
}

// ---------------------------------------------------------------------------
// Middleware: log all requests when verbose
// ---------------------------------------------------------------------------

async fn log_middleware(
    State(state): State<McpState>,
    method: Method,
    uri: Uri,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    if state.verbose {
        eprintln!("[MCP] {} {}", method, uri);
    }
    let response = next.run(request).await;
    if state.verbose {
        eprintln!("[MCP] {} {} -> {}", method, uri, response.status());
    }
    response
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

/// Wake the GUI thread so it polls the MCP channel.
fn wake_gui(state: &McpState) {
    if let Some(ctx) = state.egui_ctx.lock().unwrap().as_ref() {
        ctx.request_repaint();
    }
}

/// Build response headers (session ID, content-type).
fn response_headers(state: &McpState) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Some(ref sid) = *state.session_id.lock().unwrap() {
        headers.insert("Mcp-Session-Id", sid.parse().unwrap());
    }
    headers
}

async fn handle_post(
    State(state): State<McpState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    if state.verbose {
        eprintln!("MCP <<< {}", body);
    }

    let request: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, response_headers(&state), Json(json!({
                "jsonrpc": "2.0", "id": null,
                "error": { "code": -32700, "message": format!("Parse error: {}", e) }
            }))).into_response();
        }
    };

    let method = request.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let id = request.get("id").cloned().unwrap_or(Value::Null);

    // Validate session ID on non-initialize requests
    if method != "initialize" {
        let current_sid = state.session_id.lock().unwrap().clone();
        if let Some(ref expected) = current_sid {
            let client_sid = headers.get("mcp-session-id").and_then(|v| v.to_str().ok());
            if client_sid != Some(expected.as_str()) {
                return (StatusCode::NOT_FOUND, response_headers(&state), Json(json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": { "code": -32600, "message": "Invalid or missing session ID" }
                }))).into_response();
            }
        }
    }

    let mut resp_headers = response_headers(&state);
    let (resp_body, no_body) = match method {
        "initialize" => {
            let sid = generate_session_id();
            *state.session_id.lock().unwrap() = Some(sid.clone());
            resp_headers.insert("Mcp-Session-Id", sid.parse().unwrap());
            (handle_initialize(id, &request, &state).await, false)
        }
        "notifications/initialized" => (Value::Null, true),
        "tools/list" => (handle_tools_list(id), false),
        "tools/call" => (handle_tools_call(id, &request, &state).await, false),
        "resources/list" => (handle_resources_list(id), false),
        "resources/read" => (handle_resources_read(id, &request), false),
        _ => (json!({
            "jsonrpc": "2.0", "id": id,
            "error": { "code": -32601, "message": format!("Method not found: {}", method) }
        }), false),
    };

    if state.verbose {
        if no_body {
            eprintln!("MCP >>> (no body)");
        } else {
            eprintln!("MCP >>> {}", resp_body);
        }
    }

    if no_body {
        (StatusCode::OK, resp_headers).into_response()
    } else {
        (StatusCode::OK, resp_headers, Json(resp_body)).into_response()
    }
}

/// GET /mcp - SSE streaming (not supported, return 405)
async fn handle_get() -> Response {
    StatusCode::METHOD_NOT_ALLOWED.into_response()
}

/// DELETE /mcp - session termination (not supported, return 405)
async fn handle_delete() -> Response {
    StatusCode::METHOD_NOT_ALLOWED.into_response()
}

/// Fallback for unknown paths - return JSON 404 (not empty body)
async fn handle_fallback(method: Method, uri: Uri) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({
        "error": format!("Not found: {} {}", method, uri)
    }))).into_response()
}

// ---------------------------------------------------------------------------
// MCP method handlers
// ---------------------------------------------------------------------------

async fn handle_initialize(id: Value, request: &Value, state: &McpState) -> Value {
    let client_name = request.pointer("/params/clientInfo/name").and_then(|v| v.as_str()).unwrap_or("unknown");
    let client_version = request.pointer("/params/clientInfo/version").and_then(|v| v.as_str()).unwrap_or("?");
    eprintln!("MCP: agent connected: {} v{}", client_name, client_version);
    // Notify GUI via a msg command
    let msg = format!("msg **MCP agent connected:** {} v{}", client_name, client_version);
    let _ = send_command_str(&state.tx, &msg, state).await;
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": { "listChanged": false },
                "resources": { "subscribe": false, "listChanged": false }
            },
            "serverInfo": {
                "name": "arael-sketch",
                "version": env!("CARGO_PKG_VERSION")
            },
            "instructions": format!("{}\n\n{}", MCP_PREAMBLE, include_str!("../docs/COMMANDS.md"))
        }
    })
}

fn handle_tools_list(id: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [
                {
                    "name": "execute_command",
                    "description": "Execute a single sketch command and return the result. See instructions for available commands.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The sketch command to execute, e.g. 'add_line 0,0 5,0' or 'horizontal L0'"
                            }
                        },
                        "required": ["command"]
                    }
                },
                {
                    "name": "execute_script",
                    "description": "Execute multiple sketch commands (one per line) and return all results.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "script": {
                                "type": "string",
                                "description": "Multiple commands separated by newlines"
                            }
                        },
                        "required": ["script"]
                    }
                },
                {
                    "name": "get_sketch_state",
                    "description": "Get the current sketch state: all entities, dimensions, parameters, and constraints.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "get_help",
                    "description": "Get the full command reference documentation.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }
    })
}

async fn handle_tools_call(id: Value, request: &Value, state: &McpState) -> Value {
    let name = request.pointer("/params/name").and_then(|v| v.as_str()).unwrap_or("");
    let args = request.pointer("/params/arguments").cloned().unwrap_or(json!({}));

    match name {
        "execute_command" => {
            let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if command.is_empty() {
                return tool_error(id, "Missing 'command' argument");
            }
            let result = send_command_str(&state.tx, command, state).await;
            tool_result(id, &result)
        }
        "execute_script" => {
            let script = args.get("script").and_then(|v| v.as_str()).unwrap_or("");
            if script.is_empty() {
                return tool_error(id, "Missing 'script' argument");
            }
            let mut outputs = Vec::new();
            for line in script.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                let result = send_command_str(&state.tx, line, state).await;
                outputs.push(result);
            }
            tool_result(id, &outputs.join("\n"))
        }
        "get_sketch_state" => {
            let result = send_command_str(&state.tx, "list", state).await;
            let constraints = send_command_str(&state.tx, "list constraints", state).await;
            tool_result(id, &format!("{}\n\nConstraints:\n{}", result, constraints))
        }
        "get_help" => {
            tool_result(id, include_str!("../docs/COMMANDS.md"))
        }
        _ => {
            json!({
                "jsonrpc": "2.0", "id": id,
                "error": { "code": -32602, "message": format!("Unknown tool: {}", name) }
            })
        }
    }
}

fn handle_resources_list(id: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "resources": [
                {
                    "uri": "sketch://commands",
                    "name": "Command Reference",
                    "description": "Full sketch command documentation",
                    "mimeType": "text/markdown"
                },
                {
                    "uri": "sketch://state",
                    "name": "Sketch State",
                    "description": "Current sketch entities, constraints, and parameters",
                    "mimeType": "text/plain"
                }
            ]
        }
    })
}

fn handle_resources_read(id: Value, request: &Value) -> Value {
    let uri = request.pointer("/params/uri").and_then(|v| v.as_str()).unwrap_or("");
    match uri {
        "sketch://commands" => json!({
            "jsonrpc": "2.0", "id": id,
            "result": {
                "contents": [{
                    "uri": "sketch://commands",
                    "mimeType": "text/markdown",
                    "text": include_str!("../docs/COMMANDS.md")
                }]
            }
        }),
        _ => json!({
            "jsonrpc": "2.0", "id": id,
            "error": { "code": -32602, "message": format!("Unknown resource: {}", uri) }
        }),
    }
}

// ---------------------------------------------------------------------------
// Channel communication with GUI
// ---------------------------------------------------------------------------

const MCP_BLOCKED: &[&str] = &["save", "load", "exit", "quit"];

async fn send_command_str(tx: &mpsc::Sender<McpRequest>, command: &str, state: &McpState) -> String {
    let (resp_tx, resp_rx) = oneshot::channel();
    if tx.send(McpRequest {
        command: command.to_string(),
        response_tx: resp_tx,
        blocked_commands: MCP_BLOCKED.to_vec(),
    }).await.is_err() {
        return "Error: sketch editor disconnected".to_string();
    }
    wake_gui(state);
    resp_rx.await.unwrap_or_else(|_| "Error: no response from sketch editor".to_string())
}

fn generate_session_id() -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::time::SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn tool_result(id: Value, text: &str) -> Value {
    json!({
        "jsonrpc": "2.0", "id": id,
        "result": {
            "content": [{ "type": "text", "text": text }],
            "isError": false
        }
    })
}

fn tool_error(id: Value, text: &str) -> Value {
    json!({
        "jsonrpc": "2.0", "id": id,
        "result": {
            "content": [{ "type": "text", "text": text }],
            "isError": true
        }
    })
}

// ---------------------------------------------------------------------------
// Preamble sent before COMMANDS.md to AI agents on initialize
// ---------------------------------------------------------------------------

const MCP_PREAMBLE: &str = "You are controlling arael-sketch, a 2D CAD/drawing program, via MCP tools. \
You can create technical drawings by building geometry and constraining it. \
Use execute_command for single commands, execute_script for multi-line scripts \
(# comments and blank lines are skipped), and get_sketch_state to inspect the current sketch.";
