import os
from dotenv import load_dotenv

load_dotenv()


STOCK_CODES = [
    "4004", "4062", "5706", "6525", "6857",
    "6871", "6920", "6976", "6981", "7735",
    "285A", "9984", "5803", "166A",
    "5016", "5801", "6954", "6506", "7014",
    "3436", "6723", "7011", "7012", "7013",
    "200A", "6613", "6376", "6055", "6227",
    "7721", "4063", "1969", 
]

INDEX_CODES = [
    "^N225", # 日経
    "^NDX", # ナスダック100
    "^DJI", # ダウ30
    "^SPX", # SP500
    "^SOX", # SOX
    "USDJPY=X", # 為替
    "^VIX", #VIX
]

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")