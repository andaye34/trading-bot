import google.generativeai as genai
import yfinance as yf
import os

# Load Gemini API Key (Store it as an environment variable for security)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_market_summary():
    """
    Fetches current market conditions for major indices: S&P 500, Nasdaq, and Dow Jones.
    """
    indices = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI"
    }
    market_summary = {}
    
    for name, ticker in indices.items():
        try:
            data = yf.Ticker(ticker).history(period="20d")
            if not data.empty:
                close_price = data["Close"].iloc[-1]
                market_summary[name] = close_price
        except Exception as e:
            market_summary[name] = f"Error fetching data: {e}"
    
    return market_summary

def analyze_market_conditions():
    """
    Fetches market data and summarizes whether the market is bullish or bearish.
    """
    market_data = get_market_summary()
    summary = "Market Conditions Today:\n"
    
    for index, price in market_data.items():
        summary += f"- {index}: {price}\n"
    
    return summary

def chat_with_gemini(user_input):
    """
    Uses Google Gemini AI to analyze financial conditions and answer user queries.
    """
    market_summary = analyze_market_conditions()
    prompt = f"""
    You are a financial assistant. Here is today's market summary:
    {market_summary}
    Now, answer this user's question based on today's market data:
    {user_input}
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    
    return response.text if response else "Sorry, I couldn't fetch the response."

# Example Usage
if __name__ == "__main__":
    user_query = input("Ask about today's market: ")
    response = chat_with_gemini(user_query)
    print("\nChatbot Response:", response)
