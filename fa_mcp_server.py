# MCP Finance Server — FastMCP implementation
#
# Exposes every major MCP server primitive for the cohort demo:
#
#   TOOLS      — 6 data tools + 1 SAMPLING tool (server asks client's LLM)
#   RESOURCES  — static (watchlist, disclaimer) + templated (company profile)
#   PROMPTS    — reusable analyst prompt templates
#   LOGGING    — server→client log notifications via ctx.info/debug
#   PROGRESS   — progress notifications on the slow history fetch
#
# Deps: uv add fastmcp yfinance finnhub-python python-dotenv
# Env:  FINNHUB_KEY (optional — falls back to yfinance where possible)

import json
import os
from datetime import datetime, timedelta

import yfinance as yf
from dotenv import load_dotenv
from fastmcp import Context, FastMCP

load_dotenv()

# Server definition — name/version show up in the client's initialize() result
mcp = FastMCP(
    name="finance-analyst-server",
    version="1.0.0",
    instructions=(
        "Financial data server. Tools fetch market data; resources provide "
        "reference material (watchlist, profiles, disclaimer); prompts provide "
        "ready-made analyst task templates. One tool (summarize_news_sentiment) "
        "uses SAMPLING: it asks the client to run an LLM completion."
    ),
)

# Optional Finnhub client — the demo degrades gracefully without it.
finnhub_client = None
FINNHUB_KEY = os.environ.get("FINNHUB_KEY")
if FINNHUB_KEY:
    import finnhub
    finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)


# TOOLS
@mcp.tool()
async def get_stock_history(ticker: str, ctx: Context) -> str:
    """Gets last 7 days of historical closing prices for a stock ticker.

    Returns a JSON string with 'dates' (YYYY-MM-DD list) and 'prices' (floats).
    """
    await ctx.info(f"Fetching 7d price history for {ticker}...")  # 📢 client sees this
    try:
        history = yf.Ticker(ticker).history(period="7d")
        if history.empty:
            return json.dumps({"error": f"no history found for {ticker}"})

        await ctx.report_progress(progress=50, total=100)  # 📊 progress notification
        history.reset_index(inplace=True)
        result = {
            "ticker": ticker.upper(),
            "dates": history["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "prices": [round(p, 2) for p in history["Close"].tolist()],
        }
        await ctx.report_progress(progress=100, total=100)
        return json.dumps(result)
    except Exception as e:
        await ctx.error(f"history fetch failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_latest_quote(ticker: str, ctx: Context) -> dict:
    """Fetches the latest quote for a stock ticker.

    Returns current price, change, percent change, day high/low, open and
    previous close. Uses Finnhub when available, yfinance otherwise.

    NOTE: this returns a DICT — FastMCP automatically exposes it as MCP
    'structured content', a good contrast with the JSON-string tools above.
    """
    await ctx.info(f"Fetching latest quote for {ticker}...")
    if finnhub_client:
        try:
            return finnhub_client.quote(ticker)
        except Exception as e:
            await ctx.warning(f"finnhub quote failed, falling back: {e}")
    try:
        info = yf.Ticker(ticker).fast_info
        return {
            "c": round(info.last_price, 2),            # current
            "d": round(info.last_price - info.previous_close, 2),  # change
            "dp": round((info.last_price / info.previous_close - 1) * 100, 2),
            "h": round(info.day_high, 2),
            "l": round(info.day_low, 2),
            "o": round(info.open, 2),
            "pc": round(info.previous_close, 2),
            "_source": "yfinance",
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_company_news(ticker: str, ctx: Context) -> list:
    """Fetches the 3 most recent company news items (last 30 days).

    Each item has headline, summary, source, url and datetime.
    """
    await ctx.info(f"Fetching news for {ticker}...")
    if finnhub_client:
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            news = finnhub_client.company_news(
                ticker, _from=start.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d")
            )
            return news[:3]
        except Exception as e:
            await ctx.warning(f"finnhub news failed, falling back: {e}")
    try:
        news = yf.Ticker(ticker).news or []
        return [
            {
                "headline": n.get("title") or n.get("content", {}).get("title", ""),
                "summary": n.get("summary") or n.get("content", {}).get("summary", ""),
                "source": n.get("publisher") or n.get("content", {})
                          .get("provider", {}).get("displayName", ""),
                "datetime": n.get("providerPublishTime")
                            or n.get("content", {}).get("pubDate", ""),
            }
            for n in news[:3]
        ]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def get_recommendation_trends(ticker: str, ctx: Context) -> list:
    """Fetches the latest analyst recommendation trends.

    Returns a list of periods, most recent first, each with counts for
    strongBuy, buy, hold, sell, strongSell.
    """
    await ctx.info(f"Fetching recommendation trends for {ticker}...")
    if finnhub_client:
        try:
            return finnhub_client.recommendation_trends(ticker)
        except Exception as e:
            await ctx.warning(f"finnhub trends failed, falling back: {e}")
    try:
        recs = yf.Ticker(ticker).recommendations_summary
        if recs is None or recs.empty:
            return [{"error": "no recommendation data"}]
        rows = recs.tail(4).to_dict(orient="records")
        return [
            {
                "period": r.get("period", ""),
                "strongBuy": int(r.get("strongBuy", 0)),
                "buy": int(r.get("buy", 0)),
                "hold": int(r.get("hold", 0)),
                "sell": int(r.get("sell", 0)),
                "strongSell": int(r.get("strongSell", 0)),
            }
            for r in rows
        ]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def get_earnings_reports(ticker: str, ctx: Context) -> list:
    """Fetches the 3 most recent earnings reports (date, EPS actual/estimate).

    Requires FINNHUB_KEY — returns a helpful error otherwise.
    """
    if not finnhub_client:
        return [{"error": "FINNHUB_KEY not set — earnings tool unavailable"}]
    try:
        earnings = finnhub_client.earnings_calendar(
            _from="2025-01-01",
            to=datetime.now().strftime("%Y-%m-%d"),
            symbol=ticker,
            international=False,
        )
        if earnings and "earningsCalendar" in earnings and earnings["earningsCalendar"]:
            return sorted(
                earnings["earningsCalendar"], key=lambda x: x["date"], reverse=True
            )[:3]
        return [{"info": f"No earnings records found for {ticker} since 2025-01-01"}]
    except Exception as e:
        return [{"error": str(e)}]


# SAMPLING TOOL
#
# The SERVER asks the CLIENT to run an LLM completion via ctx.sample().
# The client controls the model/key (in our client: OpenRouter via the
# sampling_handler). Server gets back plain text.

@mcp.tool()
async def summarize_news_sentiment(ticker: str, ctx: Context) -> str:
    """Analyzes overall news sentiment for a ticker using the CLIENT's LLM.

    Demonstrates MCP SAMPLING: this tool fetches news itself, then sends a
    sampling request to the client asking its LLM to judge sentiment.
    """
    await ctx.info("📨 Sending SAMPLING request to the client's LLM...")
    news = await get_company_news(ticker, ctx)
    if not news or "error" in (news[0] if news else {}):
        return "No news available to analyze."

    headlines = "\n".join(f"- {n.get('headline', '')}" for n in news)
    result = await ctx.sample(
        messages=f"Headlines for {ticker.upper()}:\n{headlines}\n\n"
                 "Classify overall sentiment (Bullish / Neutral / Bearish) "
                 "and justify in 2 sentences.",
        system_prompt="You are a terse financial sentiment classifier.",
        max_tokens=256,
        temperature=0.0,
    )
    await ctx.info("📤 Sampling response received from client LLM.")
    return result.text

# RESOURCES — read-only reference data the client can list & read
@mcp.resource("finance://watchlist")
def watchlist() -> str:
    """The demo watchlist of tickers this server knows well."""
    return json.dumps({
        "name": "Demo Watchlist",
        "tickers": ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"],
        "updated": datetime.now().strftime("%Y-%m-%d"),
    })


@mcp.resource("finance://disclaimer")
def disclaimer() -> str:
    """Compliance disclaimer that should accompany any investment advice."""
    return (
        "DISCLAIMER: This report is generated for educational/demonstration "
        "purposes only and does not constitute financial advice. Market data "
        "may be delayed. Always consult a licensed financial advisor."
    )


# --- Templated resource: URI contains a parameter the client fills in ---
@mcp.resource("finance://profile/{ticker}")
def company_profile(ticker: str) -> str:
    """Company profile for any ticker — read as finance://profile/NVDA etc."""
    try:
        info = yf.Ticker(ticker).info
        return json.dumps({
            "ticker": ticker.upper(),
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "employees": info.get("fullTimeEmployees"),
            "summary": (info.get("longBusinessSummary") or "")[:500],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

# PROMPTS — reusable templates for clients
@mcp.prompt()
def analyst_report(ticker: str) -> str:
    """Full investment-analysis task template for a single ticker."""
    return f"""You are a Senior Finance & Investment Analyst.
Produce an investment report for **{ticker.upper()}**:
1. Gather price history, quote, news, recommendations, earnings via tools.
2. Read finance://disclaimer and include it verbatim at the end.
3. Structure: Overview → Latest News → Analyst Sentiment & Earnings →
   Investment Advice (BUY / HOLD / AVOID with bull case, bear case, risks).
Be data-driven; cite every number you use."""


@mcp.prompt()
def earnings_brief(ticker: str) -> str:
    """Short earnings-focused brief template."""
    return f"""Write a 5-bullet earnings brief for {ticker.upper()} using
get_earnings_reports and get_latest_quote. Focus on EPS surprises and
post-earnings price reaction."""


@mcp.prompt()
def compare_stocks(tickers: str) -> str:
    """Multi-ticker comparison template (comma-separated tickers)."""
    return f"""Compare these tickers: {tickers}.
For each: fetch quote + recommendation trends, then rank them by
(a) analyst sentiment and (b) 7-day momentum. End with a single pick."""


if __name__ == "__main__":
    mcp.run()  # defaults to stdio; use mcp.run(transport="http", port=8000) for HTTP
