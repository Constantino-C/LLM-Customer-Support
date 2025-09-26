from dash import Dash, html, dcc, Input, Output, State, callback_context
from assistant.infer import load, predict


# Simple global cache so we don't reload the model on every click
_cached = {"key": None, "tok": None, "model": None}


def get_model(base_model: str, adapter_path: str):
    key = (base_model, adapter_path)
    if _cached["key"] != key:
        tok, model = load(base_model, adapter_path)
        _cached.update({"key": key, "tok": tok, "model": model})
    return _cached["tok"], _cached["model"]


app = Dash(__name__)
app.title = "Customer Support AI Assistant"


app.layout = html.Div(
    style={
        "minHeight": "100vh", 
        "background": "linear-gradient(135deg, #0a192f, #112d4e)", 
        "padding": "20px",
    },
    children=[
        html.Div(
            style={
                "maxWidth": 850,
                "margin": "40px auto",
                "fontFamily": "system-ui, sans-serif",
                "padding": "20px",
                "borderRadius": "12px",
                "background": "rgba(15, 23, 42, 0.9)", 
                "color": "#f8f9fa",
                "boxShadow": "0 4px 20px rgba(0,0,0,0.3)",
            },
            children=[
        html.H1(
            "Customer Support AI Assistant",
            style={"textAlign": "center", "color": "#f1f5f9", "marginBottom": "20px"},
        ),

        html.Label("Model adapter path", style={"color": "#cbd5e1"}),
        dcc.Input(
            id="adapter",
            value="outputs/lora-adapter",
            style={
                "width": "95%",
                "marginBottom": 15,
                "padding": "10px",
                "borderRadius": "6px",
                "border": "1px solid #334155",
                "background": "#1e293b",
                "color": "#f8fafc",
            },
        ),

        html.Label("Support message", style={"color": "#cbd5e1"}),
        dcc.Textarea(
            id="message",
            placeholder="Describe customer issue here",
            style={
                "width": "95%",
                "height": 160,
                "padding": "12px",
                "borderRadius": "6px",
                "border": "1px solid #334155",
                "background": "#1e293b",
                "color": "#f8fafc",
            },
        ),

        html.Button(
            "Generate response",
            id="run",
            n_clicks=0,
            style={
                "marginTop": 15,
                "padding": "12px 20px",
                "background": "linear-gradient(90deg, #3b82f6, #2563eb)",
                "color": "white",
                "border": "none",
                "borderRadius": "6px",
                "cursor": "pointer",
                "fontWeight": "600",
            },
        ),

        html.Pre(
            id="output",
            style={
                "whiteSpace": "pre-wrap",
                "background": "#0f172a", 
                "color": "#f1f5f9",
                "padding": 16,
                "borderRadius": 8,
                "marginTop": 20,
                "height": "120px",
                "overflowY": "auto",
                "boxShadow": "inset 0 0 6px rgba(0,0,0,0.5)",
                "width": "95%"
            },
        ),

        html.Div(
            id="status",
            style={
                "marginTop": 10,
                "color": "#94a3b8",
                "fontSize": "14px",
                "fontStyle": "italic",
            },
        ),
    ],
        )
    ],
)


@app.callback(
Output("output", "children"),
Output("status", "children"),
Input("run", "n_clicks"),
State("message", "value"),
State("adapter", "value"),
prevent_initial_call=True,
)
def run_infer(n_clicks, message, adapter_path):
    if not message:
        return "", "Please enter a support message."
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok, model = get_model(base_model, adapter_path)
    res = predict(message, tok, model)
    try:
        pass
        pretty = json.dumps(json.loads(res), indent=2, ensure_ascii=False)
    except Exception:
        pretty = res
    return pretty, "Done."


if __name__ == "__main__":
    # Dash uses Flask's dev server by default: http://127.0.0.1:8050
    app.run(debug=True, host="127.0.0.1", port=8050)