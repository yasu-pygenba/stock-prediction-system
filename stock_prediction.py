import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta

from config import STOCK_CODES, INDEX_CODES

stock_list = STOCK_CODES
index_list = INDEX_CODES

# 1.データの取得
class DataAcquisition:
    """
    YahooファイナンスAPIから株価の取得と過去データへの差分更新を行うクラス
    """

    def __init__(self, stock_list: list, index_list: list, default_period: str = "2y"):
        self.stock_list = stock_list
        self.index_list = index_list
        self.default_period = default_period

    def _get_start_date_from_csv(self, file_path: str) -> str:
        """
        既存のCSVファイルから最新の日付を取得し、その翌日の日付を文字列で返す。
        ファイルが存在しない場合は None を返す。
        """
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path)
            # CSVのインデックスが日付になっている場合や、カラムにある場合に対応
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
                max_date = df['Date'].max()
            else:
                # インデックスも確認
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                max_date = df.index.max()

            if pd.isna(max_date):
                return None
            
            # 最新の翌日をスタートの日付にする
            start_date = max_date + timedelta(days=1)
            return start_date.strftime("%Y-%m-%d")
        
        except Exception as e:
            print(f"過去データの読み込みに失敗したため、全件取得します： {e}")
            return None

    def fetch_stock_data(self, file_path: str = "stock_data.csv")-> pd.DataFrame:
        """
        各銘柄のデータを取得（過去データがある場合は差分のみ取得して結合）
        """
        start_date = self._get_start_date_from_csv(file_path)

        all_stock = []
        for stock_code in self.stock_list:

            stock = yf.Ticker(f"{stock_code}.T")

            # 過去データがある場合は start を指定、ない場合は period を指定
            if start_date:
                # 今日以降の未来は指定できないように制御
                if start_date > datetime.today().strftime("%Y-%m-%d"):
                    print(f"{stock_code} はすでに最新です。")
                    continue
                stock_df = stock.history(start=start_date)
            else:
                stock_df = stock.history(period=self.default_period)

            if stock_df.empty:
                continue

            # インデックス(Date)を通常のカラムに変更
            stock_df = stock_df.reset_index()

            # PER他の取得
            info = stock.info
            stock_df['予想PER'] = info.get("forwardPE")
            stock_df['予想EPS'] = info.get("forwardEps")
            stock_df['Code'] = stock_code
            stock_df['銘柄名'] = info.get("shortName")

            all_stock.append(stock_df)
            print(stock_code, f" 取得完了（開始日：{start_date if start_date else self.default_period}）")

        print(all_stock)
        # 今回新規に取得したデータ
        if all_stock:
            new_df = pd.concat(all_stock, ignore_index=True)
        else:
            new_df = pd.DataFrame()

        # 過去データがある場合は読み込んで今回データと結合、重複箇所は削除
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            old_df = pd.read_csv(file_path)
            old_df['Date'] = pd.to_datetime(old_df['Date'])
            if not new_df.empty:
                # 日付をUTCに一度合わせてタイムゾーン情報を消去
                new_df['Date'] = pd.to_datetime(new_df['Date'], utc=True).dt.tz_localize(None)
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                combined_df = old_df
        else:
            if not new_df.empty:
                new_df["Date"] = pd.to_datetime(new_df["Date"], utc=True).dt.tz_localize(None)
            combined_df = new_df

        if not combined_df.empty:
            combined_df["Date"] = pd.to_datetime(combined_df["Date"], utc=True).dt.tz_localize(None)
            combined_df["Code"] = combined_df["Code"].astype(str)
            # 日付とCodeの組み合わせで重複があれば削除
            combined_df = combined_df.drop_duplicates(subset=['Date', 'Code'], keep='last')
            combined_df = combined_df.sort_values(by=['Code', 'Date']).reset_index(drop=True)
            # CSVへの保存
            combined_df.to_csv(file_path, index=False, encoding="utf-8-sig")

        return combined_df


    def fetch_index_data(self, file_path: str = "index_data.csv")-> pd.DataFrame:
        """
        日本・米国の指数データ取得（過去データがある場合は差分のみ取得して結合）
        """
        start_date = self._get_start_date_from_csv(file_path)

        all_index = []
        for index_code in self.index_list:
            index = yf.Ticker(index_code)

            if start_date:
                if start_date > datetime.today().strftime("%Y-%m-%d"):
                    continue
                index_df = index.history(start=start_date)
            else:
                index_df = index.history(period=self.default_period)

            if index_df.empty:
                continue
            
            index_df = index_df.reset_index()
            info = index.info
            index_df['Code'] = index_code
            index_df['銘柄名'] = info.get("shortName")

            all_index.append(index_df)
            print(f"{index_code} 取得完了（開始日：{start_date if start_date else self.default_period}）")

        if all_index:
            new_df = pd.concat(all_index, ignore_index=True)
        else:
            new_df = pd.DataFrame()

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            old_df = pd.read_csv(file_path)
            old_df['Date'] = pd.to_datetime(old_df['Date'])
            if not new_df.empty:
                # 日付をUTCに一度合わせてタイムゾーン情報を消去
                new_df['Date'] = pd.to_datetime(new_df['Date'], utc=True).dt.tz_localize(None)
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                combined_df = old_df
        else:
            if not new_df.empty:
                new_df["Date"] = pd.to_datetime(new_df["Date"], utc=True).dt.tz_localize(None)
            combined_df = new_df
        
        if not combined_df.empty:
            combined_df["Date"] = pd.to_datetime(combined_df["Date"], utc=True).dt.tz_localize(None)
            combined_df["Code"] = combined_df["Code"].astype(str)

            combined_df = combined_df.drop_duplicates(subset=['Date', 'Code'], keep='last')
            combined_df = combined_df.sort_values(by=['Code', 'Date']).reset_index(drop=True)
            combined_df.to_csv(file_path, index=False, encoding="utf-8-sig")

        return combined_df
    
# ２．前処理
class DataPreprocessor:
    """
    データ分析をするために前処理を行うクラス
    """

    def __init__(self, stock_df: pd.DataFrame, index_df: pd.DataFrame):
        self.stock_df = stock_df
        self.index_df = index_df

    def _to_number(self, df: pd.DataFrame, number_columns: list) -> pd.DataFrame:
        """文字列などの数値を数値型に変換する（共通処理）"""

        df = df.copy()
        for number_col in number_columns:
            df[number_col] = pd.to_numeric(df[number_col], errors="coerce")
        return df
    
    def process(self) -> pd.DataFrame:
        """個別株と指数の前処理、前日比率計算、指数は横結合する"""

        # -----------------------------------------------------------------------
        # １．個別株の前処理
        # -----------------------------------------------------------------------
        if "Date" not in self.stock_df.columns and self.stock_df.index.name == "Date":
            self.stock_df = self.stock_df.reset_index()
        
        self.stock_df = self.stock_df.rename(columns={
            "Date": "日付",
            "Open": "始値",
            "High": "高値",
            "Low": "安値",
            "Close": "終値",
            "Volume": "出来高",
        })

        self.stock_df["日付"] = pd.to_datetime(self.stock_df["日付"])

        number_columns = [
            "始値", "高値", "安値", "終値", "出来高"
        ]

        self.stock_df[number_columns] = self._to_number(self.stock_df[number_columns], number_columns)
        self.stock_df["Code"] = self.stock_df["Code"].astype(str)

        self.stock_df = (
            self.stock_df
            .sort_values(["Code", "日付"])
            .drop_duplicates(subset=["Code", "日付"], keep="last")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------------------------------
        # ２．指数の前処理と前日比率の計算
        # ----------------------------------------------------------------------------
        if "Date" not in self.index_df.columns and self.index_df.index.name == "Date":
            self.index_df = self.index_df.reset_index()

        self.index_df = self.index_df.rename(columns={
            "Date": "日付", "Close": "指数終値"
        })
        self.index_df["日付"] = pd.to_datetime(self.index_df["日付"])
        self.index_df['指数終値'] = pd.to_numeric(self.index_df["指数終値"], errors="coerce")
        self.index_df['Code'] = self.index_df["Code"].astype(str)

        self.index_df = (
            self.index_df
            .sort_values(["Code", "日付"])
            .drop_duplicates(subset=["Code", "日付"], keep="last")
            .reset_index(drop=True)
        )

        # 指数の前日終値と前日比率の計算 Codeごとにshiftする
        self.index_df["指数前日終値"] = self.index_df.groupby("Code")["指数終値"].shift(1)
        self.index_df["指数前日比率"] = (
            (self.index_df["指数終値"] - self.index_df["指数前日終値"])
            / self.index_df["指数前日終値"] * 100
        )

        # --------------------------------------------------------------------------------
        # ３．指数データのピボット
        # --------------------------------------------------------------------------------
        # index(横軸)に日付、columns（横軸）に「Code」、Valueに「指数前日比率」を指定
        index_pivot = self.index_df.pivot(
            index="日付",
            columns="Code",
            values="指数前日比率"
        ).reset_index()

        index_pivot.columns.name = None

        # --------------------------------------------------------------------------------
        # ４．個別株に指数の横結合（マージ）
        # --------------------------------------------------------------------------------
        merged_df = pd.merge(
            self.stock_df,
            index_pivot,
            on="日付",
            how="left"
        )

        # 指数の前日比率がNaN（欠損値）の補完は直近値、残りを0
        index_cols = [col for col in index_pivot.columns if col != "日付"]
        merged_df[index_cols] = merged_df[index_cols].ffill().fillna(0)

        return merged_df
    



if __name__ == "__main__":

    # # ----- 1.データ取得 --------------------------------------------------------
    # data_acquisition = DataAcquisition(stock_list, index_list)

    # # 各メソッドの引数に保存先CSVパスを渡し、取得・更新・結合を自動化
    # stock_df = data_acquisition.fetch_stock_data("stock_data.csv")
    # index_df = data_acquisition.fetch_index_data("index_data.csv")

    # ----- 2.前処理 --------------------------------------------------------
    # テスト用データ読み込み
    stock_df = pd.read_csv("stock_data.csv")
    index_df = pd.read_csv("index_data.csv")


    prprocessor = DataPreprocessor(stock_df, index_df)
    clean_df = prprocessor.process()

    print("\n --- 前処理後データの確認 ---")
    print("前処理完了後の形状：", clean_df.shape)

    print("\n テストおわり\n")