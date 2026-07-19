import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# 日経225採用銘柄をSTOCK_CODESへ
# CSVデータは日経指数ページより
NIKKEI_CSV_PATH = "nikkei_225_price_adjustment_factor_jp.csv"
master_df = pd.read_csv(NIKKEI_CSV_PATH, encoding="cp932", dtype={"コード": str}) # 保存形式がcp932

master_df = master_df.rename(columns={"コード": "Code"}) # Code表記へ変換
master_df = master_df.dropna(subset=["Code"]) # 欠損値行の削除

# STOCK_CODESへリスト化
STOCK_CODES = master_df["Code"].astype(str).str.strip().tolist()

# 銘柄名・業種・セクターを辞書
STOCK_NAMES = master_df.set_index("Code")["銘柄名"].to_dict()
INDUSTRY_DICT = master_df.set_index("Code")["業種"].to_dict()
SECTOR_DICT = master_df.set_index("Code")["セクター"].to_dict()

INDEX_CODES = [
    "^N225", # 日経
    "NIY=F", # 日経先物
    "^NDX", # ナスダック100
    "^DJI", # ダウ30
    "^SPX", # SP500
    "^SOX", # SOX
    "USDJPY=X", # 為替
    "^VIX", #VIX
]


DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")