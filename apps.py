import os
import yfinance as yf
import numpy as np
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

# ==========================================
# 1. The Ultimate Cloud Fix: API Hijacking
# ==========================================
# We trick CrewAI into thinking it is using OpenAI, 
# but we reroute the URL directly to Groq's ultra-fast servers.

os.environ["OPENAI_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama-3.1-8b-instant"

# Use the dedicated ChatGroq connector and pass the secret directly into it
# live_llm = ChatGroq(
#     temperature=0,
#     model_name="llama3-8b-8192",
#     api_key=st.secrets["GROQ_API_KEY"]
# )

# ==========================================
# 2. Define Custom Tools 
# ==========================================
@tool("Get Indian Stock Data")
def get_stock_data(ticker) -> str:
    """Fetches the current price and 1-month historical data for an Indian stock ticker."""
    if isinstance(ticker, dict):
        ticker = ticker.get("ticker", ticker.get("title", list(ticker.values())[0]))
    ticker_str = str(ticker).strip()

    try:
        stock = yf.Ticker(ticker_str)
        hist = stock.history(period="1mo")
        if hist.empty:
            return f"No data found for {ticker_str}."
        current_price = hist['Close'].iloc[-1]
        return f"Ticker: {ticker_str}\nCurrent Price: ₹{current_price:.2f}\nRecent Data:\n{hist.tail(5)}"
    except Exception as e:
        return f"Error fetching data for {ticker_str}: {str(e)}"

@tool("Search Indian Financial News")
def search_tool(query) -> str:
    """Search the web for recent news, articles, and commentary. Focuses on Indian markets."""
    if isinstance(query, dict):
        query = query.get("query", query.get("title", list(query.values())[0]))
    query_str = str(query).strip()

    search = DuckDuckGoSearchResults()
    localized_query = f"{query_str} NSE BSE India stock market news"
    return search.invoke(localized_query)

@tool("Predict Future Price")
def predict_stock_price(ticker) -> str:
    """Predicts the stock price for the next 5 days using a 3-month linear regression trend."""
    if isinstance(ticker, dict):
        ticker = ticker.get("ticker", ticker.get("title", list(ticker.values())[0]))
    ticker_str = str(ticker).strip()

    try:
        stock = yf.Ticker(ticker_str)
        hist = stock.history(period="3mo") 
        if hist.empty:
            return f"No data found for {ticker_str} to make a prediction."
        
        df = hist[['Close']].dropna()
        df['Day'] = np.arange(len(df))
        
        x = df['Day'].values
        y = df['Close'].values
        z = np.polyfit(x, y, 1) 
        trend_line = np.poly1d(z)
        
        last_day = x[-1]
        future_days = np.array([last_day + i for i in range(1, 6)])
        predictions = trend_line(future_days)
        
        res = f"5-Day Price Prediction for {ticker_str} (Based on 3-month mathematical linear trend):\n"
        for i, pred in enumerate(predictions):
            res += f"Day {i+1}: ₹{pred:.2f}\n"
        
        return res
    except Exception as e:
        return f"Error predicting data for {ticker_str}: {str(e)}"

# ==========================================
# 3. Main Analysis Function
# ==========================================
def run_stock_analysis(ticker_symbol):
    # --- Define Agents ---
    quant_analyst = Agent(
        role='Quantitative Analyst',
        goal=f'Analyze financial data and technical indicators for {ticker_symbol}.',
        backstory='You are a seasoned quantitative analyst on Dalal Street. You rely on historical data in INR (₹) to determine if a stock is technically sound.',
        verbose=True,
        allow_delegation=False,
        tools=[get_stock_data],
        #llm=live_llm, # <-- Secured to Groq
        max_iter=3
    )

    sentiment_analyst = Agent(
        role='Market Sentiment Analyst',
        goal=f'Gauge the public sentiment and news cycle surrounding {ticker_symbol}.',
        backstory='You are a financial journalist for an Indian business network. You look for SEBI updates, earnings, and news to determine if sentiment is bullish or bearish.',
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        #llm=live_llm, # <-- Secured to Groq
        max_iter=3
    )

    price_predictor = Agent(
        role='Algorithmic Forecaster',
        goal=f'Generate a 5-day price forecast for {ticker_symbol} using mathematical models.',
        backstory='You are a data scientist who builds predictive models for Indian stocks. You rely strictly on the outputs of your linear regression tool to map out short-term price targets.',
        verbose=True,
        allow_delegation=False,
        tools=[predict_stock_price],
       # llm=live_llm, # <-- Secured to Groq
        max_iter=3
    )

    strategist = Agent(
        role='Chief Investment Strategist',
        goal=f'Synthesize data, sentiment, and forecasts into a final brief for {ticker_symbol}.',
        backstory='You are an elite fund manager in India. You compile reports from your analysts and forecaster to write executive summaries with a clear verdict (Bullish, Bearish, Neutral).',
        verbose=True,
        allow_delegation=False, 
       # llm=live_llm, # <-- Secured to Groq
        max_iter=3
    )

    # --- Define Tasks ---
    task_quant = Task(
        description=f'Fetch recent price history for {ticker_symbol}. Summarize the price action in INR (₹).',
        expected_output='A report detailing current price and recent movement.',
        agent=quant_analyst
    )

    task_sentiment = Task(
        description=f'Search for recent news regarding {ticker_symbol}. Summarize the Indian market sentiment.',
        expected_output='A summary of news and a conclusion on market sentiment.',
        agent=sentiment_analyst
    )

    task_predict = Task(
        description=f'Use your mathematical tool to calculate the next 5 days of prices for {ticker_symbol}. Format the output clearly.',
        expected_output='A list of estimated prices for the next 5 trading days based on the mathematical trend.',
        agent=price_predictor
    )

    task_synthesis = Task(
        description=f'Review the quant analysis, sentiment analysis, AND the 5-day forecast for {ticker_symbol}. Write a final investment brief containing: 1. Executive Summary, 2. Technical Outlook, 3. Sentiment Outlook, 4. 5-Day Price Forecast, 5. Final Verdict.',
        expected_output='A professional investment brief formatted in Markdown.',
        agent=strategist
    )

    # --- Assemble Crew ---
    stock_analysis_crew = Crew(
        agents=[quant_analyst, sentiment_analyst, price_predictor, strategist], 
        tasks=[task_quant, task_sentiment, task_predict, task_synthesis],     
        process=Process.sequential
    )

    result = stock_analysis_crew.kickoff()
    return str(result)

# ==========================================
# 4. Streamlit UI Build
# ==========================================
st.set_page_config(page_title="Dalal Street AI", page_icon="📈", layout="wide")

st.title("📈 Dalal Street AI Analyst")
st.write("Enter an Indian stock ticker to generate a comprehensive fundamental, technical, and predictive report.")

ticker_input = st.text_input("Enter Ticker Symbol (e.g., RELIANCE.NS, TCS.NS, INFY.BO):", "RELIANCE.NS")

if st.button("Generate Analysis"):
    if ticker_input:
        with st.spinner(f"Your team of 4 AI agents is analyzing and forecasting {ticker_input}..."):
            try:
                final_report = run_stock_analysis(ticker_input)
                st.success("Analysis Complete!")
                st.markdown("---")
                st.markdown(final_report)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid ticker symbol.")






