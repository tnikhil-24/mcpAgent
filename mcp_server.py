from fastmcp import FastMCP
from pandas import DataFrame
import yfinance as yf

mcp = FastMCP("stocks_server")

@mcp.tool()
def fetch_stock_info(symbol: str) -> dict:
    """Get Company's general information."""
    stock = yf.Ticker(symbol)
    return stock.info # dictionary containing market cap, companies name, address

@mcp.tool()
def fetch_quarterly_financials(symbol: str) -> DataFrame :
    """Get stock quarterly financials."""
    stock = yf.Ticker(symbol)
    return stock.quarterly_financials.T

@mcp.tool()
def fetch_annual_financials(symbol: str) -> DataFrame:
    """Get stock annual financials."""
    stock = yf.Ticker(symbol)
    return stock.financials.T

if __name__ == "__main__":
    mcp.run(transport="stdio")