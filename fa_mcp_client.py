# Finance Advisor — MCP Client + HF Smol Agent
#
# Demonstrates, in order:
#   1. MCP connection (stdio transport)
#   2. Capability discovery  → list_tools() / list_prompts() / list_resources()
#   3. Tool usage            → discovered server tools handed to a CodeAgent
#   4. Sampling (client side) → server can ask THIS client for an LLM completion
#   5. Client-side rendering  → local plotly tools the agent "discovers" too
#   6. Gradio UI              → financial report with charts + narrative
#
# The client is intentionally "dumb": nothing about the finance domain is
# hard-coded into the agent. The workflow emerges from the tool descriptions
# discovered at runtime.

import json
import os
from datetime import datetime

import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI
from plotly.subplots import make_subplots

from smolagents import CodeAgent, MCPClient, OpenAIServerModel, tool
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

load_dotenv()

# Configuration
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise ValueError("Please set the API_KEY environment variable.")

THE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free" #"nvidia/nemotron-nano-9b-v2:free"
SERVER_PARAMS = StdioServerParameters(command="python", args=["fa_mcp_server.py"])

# Raw OpenAI client — used ONLY for the sampling demo (server-initiated LLM
# calls that the client fulfills on the server's behalf).
_sampling_llm = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# smolagents model wrapper — powers the agent loop.
model = OpenAIServerModel(
    model_id=THE_MODEL,
    api_base="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    temperature=0.0,
)

# PART 1 — MCP CAPABILITY DISCOVERY (tools / prompts / resources)
#
# This uses a RAW MCP ClientSession so the cohort can see the actual protocol
# methods. smolagents' MCPClient only surfaces tools, so we do discovery
# manually and log everything.
async def discover_capabilities() -> str:
    """Connect to the server, enumerate everything it exposes, return a log."""
    log = ["## MCP Discovery Log\n"]

    async with AsyncExitStack() as stack:
        read_pipe, write_pipe = await stack.enter_async_context(stdio_client(SERVER_PARAMS))
        session = await stack.enter_async_context(
            ClientSession(read_pipe, write_pipe, sampling_callback=sampling_handler)
        )
        init_result = await session.initialize()

        log.append(f"**Connected to:** `{init_result.serverInfo.name}` "
                   f"v{init_result.serverInfo.version}")
        log.append(f"**Protocol version:** `{init_result.protocolVersion}`")
        log.append(f"**Server capabilities:** `{init_result.capabilities}`\n")

        # --- TOOLS ---
        tools_resp = await session.list_tools()
        log.append(f"### Tools discovered: {len(tools_resp.tools)}")
        for t in tools_resp.tools:
            params = list((t.inputSchema or {}).get("properties", {}).keys())
            log.append(f"- `{t.name}({', '.join(params)})` — {t.description}")

        # --- PROMPTS ---
        try:
            prompts_resp = await session.list_prompts()
            log.append(f"\n### Prompts discovered: {len(prompts_resp.prompts)}")
            for p in prompts_resp.prompts:
                args = [a.name for a in (p.arguments or [])]
                log.append(f"- `{p.name}({', '.join(args)})` — {p.description}")
        except Exception as e:
            log.append(f"\n### Prompts: not supported by server ({e})")

        # --- RESOURCES ---
        try:
            res_resp = await session.list_resources()
            log.append(f"\n### Resources discovered: {len(res_resp.resources)}")
            for r in res_resp.resources:
                log.append(f"- `{r.uri}` — {r.name}: {r.description}")
                # Optionally read one to show the flow:
                # content = await session.read_resource(r.uri)
        except Exception as e:
            log.append(f"\n### Resources: not supported by server ({e})")

    return "\n".join(log)


# PART 2 — SAMPLING (client-side handler)
#
# Sampling is server → client: the SERVER asks the CLIENT to run an LLM
# completion (e.g. "summarize this news"). The client stays in control of
# which model/key is used. This handler only fires if the server calls
# session.create_message() — include a tool on your server that does so
# to demo it live.

async def sampling_handler(context, params: types.CreateMessageRequestParams):
    """Fulfills a server-initiated sampling request using OpenRouter."""
    print(f"\n SAMPLING REQUEST from server! system_prompt={params.systemPrompt!r}")

    messages = []
    if params.systemPrompt:
        messages.append({"role": "system", "content": params.systemPrompt})
    for m in params.messages:
        messages.append({"role": m.role, "content": m.content.text})

    resp = _sampling_llm.chat.completions.create(
        model=MODEL_ID, messages=messages, temperature=0.0,
        max_tokens=params.maxTokens or 512,
    )
    text = resp.choices[0].message.content
    print(f" SAMPLING RESPONSE sent back to server ({len(text)} chars)")

    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text=text),
        model=MODEL_ID,
        stopReason="endTurn",
    )

# PART 3 — LOCAL VISUALIZATION TOOLS
#
# These live on the CLIENT. The agent discovers them alongside server tools,
# so the LLM decides when to chart what. Figures are captured in a global
# list for Gradio to render afterwards.
captured_figures: list[go.Figure] = []


@tool
def render_price_chart(dates: list, prices: list) -> str:
    """Renders a price-history line chart (running trend of a stock ticker).

    Args:
        dates: List of date strings in YYYY-MM-DD format.
        prices: List of closing prices (floats) aligned with dates.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines+markers", name="Close Price",
        line=dict(color="#00d4aa", width=3), marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.1)",
    ))
    fig.update_layout(
        title="Price History (7d)", template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified",
    )
    captured_figures.append(fig)
    return "Price chart rendered and attached to the report."


@tool
def render_recommendations_chart(strongBuy: int, buy: int, hold: int,
                                 sell: int, strongSell: int) -> str:
    """Renders an analyst-recommendation (sentiment) bar chart.

    Args:
        strongBuy: Count of strong-buy ratings.
        buy: Count of buy ratings.
        hold: Count of hold ratings.
        sell: Count of sell ratings.
        strongSell: Count of strong-sell ratings.
    """
    labels = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
    values = [strongSell, sell, hold, buy, strongBuy]
    colors = ["#ff4757", "#ff6b81", "#ffa502", "#7bed9f", "#2ed573"]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=values, textposition="outside"))
    fig.update_layout(
        title="Analyst Sentiment", template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        xaxis_title="Rating", yaxis_title="Count",
    )
    captured_figures.append(fig)
    return " Sentiment chart rendered and attached to the report."


LOCAL_TOOLS = [render_price_chart, render_recommendations_chart]

# PART 4 — THE AGENT RUN
#
# MCPClient connects, discovers tools, and adapts them into smolagents Tool
# objects. We concatenate server tools + local viz tools — the agent sees one
# flat toolbox and figures out the workflow itself. The "plan" lives entirely
# in the task prompt below, NOT on the server.
TASK_TEMPLATE = """You are a Senior Finance & Investment Analyst.

Produce a professional investment report for the ticker **{ticker}**.

## STRICT OUTPUT RULES (violating these crashes the run)
- Every step must be VALID PYTHON ONLY. Never write markdown, headings, or
  prose outside of Python strings or comments.
- NEVER invent prices, news, or numbers. Use ONLY values returned by tools.
- The report is delivered EXCLUSIVELY by calling final_answer(report) where
  report is a Python string variable containing the markdown.

## Data contract (exact types — follow precisely)
- ALWAYS start your code with: import json
- get_stock_history(ticker) -> JSON STRING. Parse it:
      history = json.loads(get_stock_history(ticker))
      # history is now {"dates": [...], "prices": [...]}
- get_latest_quote(ticker) -> DICT directly. quote["c"] = current price,
  quote["d"] = change, quote["dp"] = percent change. Do NOT json.loads it.
- get_company_news(ticker) -> LIST of dicts. news[0]["headline"], news[0]["summary"]
- get_recommendation_trends(ticker) -> LIST of dicts. latest = trends[0],
  then latest["strongBuy"], latest["buy"], latest["hold"], latest["sell"], latest["strongSell"]
- get_earnings_reports(ticker) -> LIST of dicts, may contain an "info"/"error"
  note instead of data. Wrap in try/except; if it fails, set earnings = "unavailable".

## Workflow
Step 1 — GATHER: call get_stock_history, get_latest_quote, get_company_news,
get_recommendation_trends, get_earnings_reports for "{ticker}". json.loads()
any JSON-string results. print() what you got.

Step 2 — VISUALIZE:
- render_price_chart(dates=history["dates"], prices=history["prices"])
- render_recommendations_chart(strongBuy=latest["strongBuy"], buy=latest["buy"],
    hold=latest["hold"], sell=latest["sell"], strongSell=latest["strongSell"])

Step 3 — REPORT: build ONE markdown string variable `report` with sections:
## Overview  (current price, 7-day trend, company activity)
## Latest News  (headlines + one-line takeaways)
## Analyst Sentiment & Earnings
## Investment Advice  (BUY / HOLD / AVOID + bull case, bear case, key risks —
cite the actual numbers you fetched)

Then finish with: final_answer(report)
"""

def run_analysis(ticker: str) -> tuple[go.Figure | None, str, str, str]:
    global captured_figures
    captured_figures = []
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None, "Please enter a ticker symbol.", "", ""

    with MCPClient(SERVER_PARAMS, structured_output=True) as server_tools:
        print(f"MCPClient discovered {len(server_tools)} server tools: "
              f"{[t.name for t in server_tools]}")

        agent = CodeAgent(
            tools=[*server_tools, *LOCAL_TOOLS],
            model=model,
            max_steps=15,
            additional_authorized_imports=["json"],
            verbosity_level=1,  # prints the agent's code steps — great for demos
        )
        report = agent.run(TASK_TEMPLATE.replace("{ticker}", ticker))

    # Combine captured figures into one stacked dashboard.
    final_plot = None
    if len(captured_figures) > 1:
        final_plot = make_subplots(
            rows=len(captured_figures), cols=1,
            subplot_titles=[f.layout.title.text for f in captured_figures],
            vertical_spacing=0.12,
        )
        for i, fig in enumerate(captured_figures, 1):
            for trace in fig.data:
                final_plot.add_trace(trace, row=i, col=1)
        final_plot.update_layout(height=420 * len(captured_figures),
                                 template="plotly_dark",
                                 paper_bgcolor="#1a1a2e",
                                 plot_bgcolor="#1a1a2e", showlegend=False)
    elif captured_figures:
        final_plot = captured_figures[0]

    return final_plot, f" Analysis complete for {ticker}", str(report), ""


async def analyze_stock(ticker: str):
    """Gradio entrypoint: discovery log first, then the agent run."""
    discovery_log = await discover_capabilities()
    plot, status, report, _ = run_analysis(ticker)
    return plot, status, report, discovery_log


# PART 5 — GRADIO UI
with gr.Blocks(theme=gr.themes.Base(primary_hue="teal"),
               title="MCP Investment Analyst") as demo:
    gr.Markdown("# MCP Investment Analyst\n"
                "> Client discovers server capabilities at runtime — nothing is hard-coded.")

    with gr.Row():
        ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="NVDA, AAPL, TSLA...",
                                  scale=3, value="NVDA")
        analyze_button = gr.Button(" Analyze", variant="primary", scale=1)

    status_output = gr.Textbox(label="Status", interactive=False)
    plot_output = gr.Plot(label="Visual Report")
    summary_output = gr.Markdown(label="Financial Report")

    with gr.Accordion(" MCP Discovery Log (tools / prompts / resources)", open=False):
        discovery_output = gr.Markdown()

    analyze_button.click(
        analyze_stock, inputs=[ticker_input],
        outputs=[plot_output, status_output, summary_output, discovery_output],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue="teal"))
