from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool
from dotenv import load_dotenv
import os
import requests


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@function_tool
def get_all_coin_prices() -> str:
    url = "https://api.coinlore.net/api/tickers/"
    response = requests.get(url)
    if response.status_code == 200:
        coins = response.json()["data"][:10]
        return ([f"{coin['symbol']} (id: {coin['id']}): ${coin['price_usd']}" for coin in coins])
    else:
        return "Failed to fetch coin prices."


@function_tool
def get_coin_price_by_id(id: str) -> str:
    url = f"https://api.coinlore.net/api/ticker/?id={id}"
    response = requests.get(url)
    if response.status_code == 200:
        coin = response.json()[0]
        return f"{coin['name']} ({coin['symbol']}) current price is ${coin['price_usd']}"
    else:
        return "Failed to fetch coin price."


crypto_data_agent= Agent(
    name="Crypto agent",
   instructions=(
    "You are a crypto assistant. Use tools to fetch real-time prices for cryptocurrencies. "
    "Use 'get_all_coin_prices' to get all prices, or 'get_coin_price_by_id' to get one coin's price by id."
),
    model=model,
    tools=[get_all_coin_prices, get_coin_price_by_id],
)

def run_crypto_checker():
    print("💹 Real-Time Crypto Price Checker")
    print("1. Get All Coin Prices")
    print("2. Get Price of Specific Coin")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        prompt = "Get me all coin prices."
    elif choice == "2":
        symbol_id = input("Enter coin ID (e.g., 90 for BTC, 80 for ETH): ").strip()
        prompt = f"What is the price of the coin with ID {symbol_id}?"
    else:
        print("Invalid choice.")
        return

    result = Runner.run_sync(
        crypto_data_agent,
        input=prompt,
        run_config=config
    )

    print("\n📊 Result:")
    print(result.final_output)


run_crypto_checker()
