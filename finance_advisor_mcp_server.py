import finnhub
import os
from mcp.server.fastmcp import FastMCP 
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("Please set the FINNHUB_API_KEY environment variable.")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

mcp = FastMCP("IK-Week03-MCP-server")

# --- The "Plan"
@mcp.prompt()
def analyze_stock(ticker: str) -> str:
    """
    Returns the system prompt and instructions for analyzing a specific stock.
    This dictates the agent's behavior and workflow.
    """
    return f"""
    You are a Financial Market Data Server for publicly traded equities.
    You expose these tools, each requiring a `ticker` (string, required):
    1. Data Gathering - For ALL tools below, you MUST pass ticker="{ticker}":
       - Call `get_stock_history(ticker="{ticker}")` to get recent price history (e.g., last ~7 trading days of OHLCV/time-series data)..
       - Call `get_latest_quote(ticker: "{ticker}")` the latest quote (e.g., last price, bid/ask, sizes, timestamp).
       - Call `get_recommendation_trends(ticker="{ticker}")` to gather analyst recommendation data (e.g., buy/hold/sell counts, consensus rating, targets).
       - Call `get_company_news(ticker="{ticker}")` to gain insights from recent company news items (e.g., id, headline, source, published_at, url, metadata)
       - Call `get_earnings_reports(ticker="{ticker}")` to get recent earnings data (e.g., fiscal period, EPS, revenue, surprise, event time, transcript/filing links).

    Behavior rules:
    - Only call a tool when the client explicitly requests that tool by name.
    - Do not guess which tool to use; ask the client to clarify if unclear.
    - Validate `ticker` is present and non-empty before calling any tool; otherwise return a structured parameter error.
    - Always return raw, structured JSON as provided by upstream sources.
    - Do not summarize, explain, interpret, visualize, or format results beyond valid JSON.
    - Do not perform orchestration or planning; the client is responsible for combining, analyzing, and presenting data.

    The client is responsible for planning, orchestration, and presentation.
    """

# --- The "Tools"
@mcp.tool()
def get_stock_history(ticker: str) -> str:
    """
    Gets last 7 days of historical price for a given stock ticker.
    Returns JSON string with 'dates' and 'prices'.
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="7d")
        if history.empty:
            return '{"error": "no history found"}'

        history.reset_index(inplace=True)
        dates = history['Date'].dt.strftime('%Y-%m-%d').tolist()
        prices = history['Close'].tolist()
        import json
        return json.dumps({"dates": dates, "prices": prices})
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'
        
@mcp.tool()
def get_latest_quote(ticker: str) -> dict:
    """Fetches the latest quote for a given stock ticker."""
    try:
        return finnhub_client.quote(ticker)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_company_news(ticker: str) -> list:
    """Fetches recent company news (last 30 days)."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        news = finnhub_client.company_news(ticker, _from=start_date.strftime("%Y-%m-%d"), to=end_date.strftime("%Y-%m-%d"))
        return news[:3]
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_recommendation_trends(ticker: str) -> list:
    """Fetches the latest analyst recommendation trends."""
    try:
        trends = finnhub_client.recommendation_trends(ticker)
        return trends
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_earnings_reports(ticker: str) -> list:
    """Fetches the top 3 most recent earnings call transcripts."""
    try:
        earnings = finnhub_client.earnings_calendar(_from="2025-01-01", to=datetime.now().strftime("%Y-%m-%d"), symbol=ticker, international=False)
        if earnings and 'earningsCalendar' in earnings:
            return sorted(earnings['earningsCalendar'], key=lambda x: x['date'], reverse=True)[:3]
        else:
            return []
    except Exception as e:
        return [{"error": str(e)}]

if __name__ == "__main__":
    mcp.run(transport="stdio")