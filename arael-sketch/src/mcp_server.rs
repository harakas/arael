// MCP (Model Context Protocol) server embedded in the sketch editor.
// Runs an async HTTP server (axum/tokio) in a background thread.
// Communicates with the GUI thread via channels.
//
// Protocol: JSON-RPC 2.0 over HTTP POST on /mcp
// Spec: https://modelcontextprotocol.io/specification/

#![cfg(not(target_arch = "wasm32"))]

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
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
}

/// Start the MCP server on the given address.
/// Returns a receiver for commands that the GUI thread should poll.
pub fn start(addr: SocketAddr, verbose: bool) -> mpsc::Receiver<McpRequest> {
    let (tx, rx) = mpsc::channel::<McpRequest>(32);
    let state = McpState { tx: Arc::new(tx), verbose };
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new()
            .expect("Failed to create tokio runtime for MCP server");
        rt.block_on(async move {
            let app = Router::new()
                .route("/mcp", post(handle_post))
                .with_state(state);
            let listener = tokio::net::TcpListener::bind(addr).await
                .unwrap_or_else(|e| panic!("MCP server failed to bind {}: {}", addr, e));
            eprintln!("MCP server listening on http://{}/mcp", addr);
            axum::serve(listener, app).await.unwrap();
        });
    });
    rx
}

async fn handle_post(
    State(state): State<McpState>,
    _headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    if state.verbose {
        eprintln!("MCP <<< {}", body);
    }
    let request_tx = &state.tx;
    let request: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, Json(json!({
                "jsonrpc": "2.0", "id": null,
                "error": { "code": -32700, "message": format!("Parse error: {}", e) }
            }))).into_response();
        }
    };

    let method = request.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let id = request.get("id").cloned().unwrap_or(Value::Null);

    let response = match method {
        "initialize" => handle_initialize(id, &request, request_tx).await.into_response(),
        "notifications/initialized" => StatusCode::OK.into_response(),
        "tools/list" => handle_tools_list(id).into_response(),
        "tools/call" => handle_tools_call(id, &request, request_tx).await.into_response(),
        "resources/list" => handle_resources_list(id).into_response(),
        "resources/read" => handle_resources_read(id, &request).into_response(),
        _ => (StatusCode::OK, Json(json!({
            "jsonrpc": "2.0", "id": id,
            "error": { "code": -32601, "message": format!("Method not found: {}", method) }
        }))).into_response(),
    };
    if state.verbose {
        // Log response body (extract from response is complex, just log method)
        eprintln!("MCP >>> responded to: {}", method);
    }
    response
}

async fn handle_initialize(id: Value, request: &Value, request_tx: &mpsc::Sender<McpRequest>) -> impl IntoResponse {
    let client_name = request.pointer("/params/clientInfo/name").and_then(|v| v.as_str()).unwrap_or("unknown");
    let client_version = request.pointer("/params/clientInfo/version").and_then(|v| v.as_str()).unwrap_or("?");
    eprintln!("MCP: agent connected: {} v{}", client_name, client_version);
    // Notify GUI via a msg command
    let msg = format!("msg **MCP agent connected:** {} v{}", client_name, client_version);
    let _ = send_command(request_tx, &msg).await;
    Json(json!({
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
            "instructions": MCP_INSTRUCTIONS
        }
    }))
}

fn handle_tools_list(id: Value) -> impl IntoResponse {
    Json(json!({
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
    }))
}

async fn handle_tools_call(id: Value, request: &Value, request_tx: &mpsc::Sender<McpRequest>) -> impl IntoResponse {
    let name = request.pointer("/params/name").and_then(|v| v.as_str()).unwrap_or("");
    let args = request.pointer("/params/arguments").cloned().unwrap_or(json!({}));

    match name {
        "execute_command" => {
            let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if command.is_empty() {
                return Json(tool_error(id, "Missing 'command' argument"));
            }
            let result = send_command(request_tx, command).await;
            Json(tool_result(id, &result))
        }
        "execute_script" => {
            let script = args.get("script").and_then(|v| v.as_str()).unwrap_or("");
            if script.is_empty() {
                return Json(tool_error(id, "Missing 'script' argument"));
            }
            // Execute each line as a separate command, collect results
            let mut outputs = Vec::new();
            for line in script.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                let result = send_command(request_tx, line).await;
                outputs.push(result);
            }
            Json(tool_result(id, &outputs.join("\n")))
        }
        "get_sketch_state" => {
            let result = send_command(request_tx, "list").await;
            let constraints = send_command(request_tx, "list constraints").await;
            Json(tool_result(id, &format!("{}\n\nConstraints:\n{}", result, constraints)))
        }
        "get_help" => {
            Json(tool_result(id, include_str!("../docs/COMMANDS.md")))
        }
        _ => {
            Json(json!({
                "jsonrpc": "2.0", "id": id,
                "error": { "code": -32602, "message": format!("Unknown tool: {}", name) }
            }))
        }
    }
}

fn handle_resources_list(id: Value) -> impl IntoResponse {
    Json(json!({
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
    }))
}

fn handle_resources_read(id: Value, request: &Value) -> impl IntoResponse {
    let uri = request.pointer("/params/uri").and_then(|v| v.as_str()).unwrap_or("");
    match uri {
        "sketch://commands" => Json(json!({
            "jsonrpc": "2.0", "id": id,
            "result": {
                "contents": [{
                    "uri": "sketch://commands",
                    "mimeType": "text/markdown",
                    "text": include_str!("../docs/COMMANDS.md")
                }]
            }
        })),
        _ => Json(json!({
            "jsonrpc": "2.0", "id": id,
            "error": { "code": -32602, "message": format!("Unknown resource: {}", uri) }
        })),
    }
}

const MCP_BLOCKED: &[&str] = &["save", "load"];

async fn send_command(tx: &mpsc::Sender<McpRequest>, command: &str) -> String {
    let (resp_tx, resp_rx) = oneshot::channel();
    if tx.send(McpRequest {
        command: command.to_string(),
        response_tx: resp_tx,
        blocked_commands: MCP_BLOCKED.to_vec(),
    }).await.is_err() {
        return "Error: sketch editor disconnected".to_string();
    }
    resp_rx.await.unwrap_or_else(|_| "Error: no response from sketch editor".to_string())
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
// Condensed instructions sent to AI agents on initialize
// ---------------------------------------------------------------------------

const MCP_INSTRUCTIONS: &str = r#"You are controlling arael-sketch, a 2D parametric constraint-based sketch editor.

ENTITIES: Lines (L0, L1...), Points (P0...), Arcs/Circles (A0...), Dimensions (d0...).
Endpoints: L0.p1, L0.p2, A0.center, A0.start, A0.end.

COORDINATES: x,y | @dx,dy (relative to cursor) | cursor | L0.p2 | midpoint(L0)
EXPRESSIONS: Entity properties (L0.length, A0.radius), params, math (sqrt, sin, pi).

GEOMETRY: add_line x1,y1 x2,y2 | add_point x,y | add_circle cx,cy r | add_arc x1,y1 x2,y2 xm,ym | offset_line L0 dist
Line chaining: add_line @dx,dy (from cursor). Auto-coincident at matching endpoints (noconnect to suppress).

CAPTURE: _ = last entity. name = add_line ... to name it. Use names in commands: horizontal base.

CONSTRAINTS: horizontal L0 | vertical L0 | parallel L0 L1 | perpendicular L0 L1 | equal L0 L1 | collinear L0 L1 | tangent L0 A0 | coincident L0.p2 L1.p1 | concentric A0 A1 | midpoint P0 L0 | symmetry L0 L1 L2 | symmetry P0 L0 P1 | symmetry L0.p1 L1 L2.p1 | point_on P0 L0

DIMENSIONS: length L0 5 | length L0 "expr" | radius A0 1.5 | angle L0 L1 45 | distance L0.p1 L1.p2 3
DERIVED DIMS: Append 'derived' to create display-only dims: length L0 derived | radius A0 1.5 derived. Toggle: set_derived d0 | set_driven d0 [value]

PARAMS: param name value | param name "expr" | del_param name | rename_param old new

INTROSPECTION: info L0 | list | list constraints | print expr | find x,y [r] | dof | cost

STYLE: style L0 solid|dashed|dashdot
CURSOR: cursor x,y | cursor @dx,dy | cursor off
LOCK: lock L0.p1 | unlock L0.p1
HISTORY: undo [n] | redo [n] | history
FILE: save path.json | load path.json | clear
REMOVE: delete L0 | remove_dim d0 | remove_constraint L0 horizontal | remove_constraint L0 L1 parallel

GEO FUNCTIONS: intersect(L0,L1) | midpoint(L0) | project(P0,L0) | along(L0,0.5) | tangent(L0) | normal(L0) | rotate(P0,center,angle) | mirror(P0,L0) | dist(P0,P1) | dist(P0,L0) | angle(L0,L1)

Use execute_command for single commands. Use execute_script for multi-line scripts (# comments, blank lines skipped). Use get_sketch_state to see current state. Use get_help for full documentation.
"#;
