import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta

import logging
import requests
from config import DISCORD_WEBHOOK_URL
from config import STOCK_CODES, INDEX_CODES, STOCK_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def __init__(self, stock_df: pd.DataFrame, index_df: pd.DataFrame, stock_names: dict = STOCK_NAMES):
        self.stock_df = stock_df
        self.index_df = index_df
        self.stock_names = stock_names

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

        # 銘柄名をCodeに合わせて変換
        self.stock_df["銘柄名"] = self.stock_df["Code"].astype(str).map(STOCK_NAMES)

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
    
# ３．特徴量生成
class FeatureEngineer:

    def __init__(self, index_list: list):
        """
        configから指数のリストを受け取る（粗結合を保ち、内部での未定義エラーを防ぐ）
        """
        self.index_list = index_list

    def create_features(self, clean_df: pd.DataFrame) -> pd.DataFrame:

        df = clean_df.copy()

        # 前日終値
        df["前日終値"] = df.groupby("Code")["終値"].shift(1)
        df["前日始値"] = df.groupby("Code")["始値"].shift(1)

        # MAの計算
        df["MA5"] = (
            df.groupby("Code")["前日終値"]
            .transform(lambda x: x.rolling(5).mean())
        )

        df["MA25"] = (
            df.groupby("Code")["前日終値"]
            .transform(lambda x: x.rolling(25).mean())
        )

        df["STD25"] = (
            df.groupby("Code")["前日終値"]
            .transform(lambda x: x.rolling(25).std(ddof=0))
        )

        for i in range(1, 4):
            df[f"+{i}σ"] = df["MA25"] + df["STD25"] * i
            df[f"-{i}σ"] = df["MA25"] - df["STD25"] * i

        
        # GU/GD他の計算
        df["前日寄り率"] = (
            (df["始値"] - df["前日終値"])
            / df["前日終値"] * 100
        )

        df["GU_GD"] = np.select(
            [
                df["始値"] >= df["前日終値"] * 1.01,
                df["始値"] <= df["前日終値"] * 0.99,
            ],
            ["GU", "GD"],
            default="レンジ"
        )

        df["MA5向き"] = (
            df.groupby("Code")["MA5"]
            .diff()
            .apply(lambda x: "上向き" if x > 0 else "下向き" if x < 0 else None)
        )

        df["MA25向き"] = (
            df.groupby("Code")["MA25"]
            .diff()
            .apply(lambda x: "上向き" if x > 0 else "下向き" if x < 0 else None)
        )

        df["前日終値_MA5判定"] = np.select(
            [
                df["前日終値"] > df["MA5"],
                df["前日終値"] < df["MA5"],
            ],
            ["MA5より上", "MA5より下"],
            default=None
        )

        df["前日終値_MA25判定"] = np.select(
            [
                df["前日終値"] > df["MA25"],
                df["前日終値"] < df["MA25"],
            ],
            ["MA25より上", "MA25より下"],
            default=None
        )

        df["前日終値_MA5乖離率"] = (
            (df["前日終値"] - df["MA5"])
            / df["MA5"] * 100
        )

        df["前日終値_MA25乖離率"] = (
            (df["前日終値"] - df["MA25"])
            / df["MA25"] * 100
        )

        df["引買_寄売_損益額_1株"] = df["始値"] - df["前日終値"]
        df["引買_寄売_損益率"] = (
            df["引買_寄売_損益額_1株"] / df["前日終値"] * 100
        )

        df["寄買_引売_損益額_1株"] = df["終値"] - df["始値"]
        df["寄買_引売_損益率"] = (
            df["寄買_引売_損益額_1株"] / df["始値"] * 100
        )

        # 前日騰落率
        df["前日騰落率"] = (
            (df["終値"] - df["前日終値"]) / df["前日終値"] * 100
        )

        # 前日値幅率
        df["値幅率"] = (
            (df["高値"] - df["安値"]) / df["終値"] * 100
        )

        # 実体率
        df["実体率"] = (
            (df["終値"] - df["始値"]) / df["始値"] * 100
        )

        # 上ヒゲ率
        df["上ヒゲ率"] = (
            (df["高値"] - df[["始値", "終値"]].max(axis=1))
            / df["始値"] * 100
        )

        # 下ヒゲ率
        df["下ヒゲ率"] = (
            (df[["始値", "終値"]].min(axis=1) - df["安値"])
            / df["始値"] * 100
        )

        # 出来高25日平均
        df["出来高MA25"] = (
            df.groupby("Code")["出来高"]
            .transform(lambda x: x.rolling(25).mean())
        )

        # 出来高急増率
        df["出来高倍率25"] = df["出来高"] / df["出来高MA25"]

        # 念のため再度日付順に並び替え
        df = df.sort_values(["Code", "日付"]).reset_index(drop=True)

        # ==========================
        # target：当日寄り買い、引け売りで利益が出たか
        # ==========================
        df["target"] = (
            df["終値"] > df["始値"]
        ).astype(int)

        # ==========================
        # 個別銘柄：前日情報にずらす
        # その日のデータはまだ存在していなため
        # 前日の情報をスライドさせる
        # ==========================
        stock_shift_cols = [
            "終値",
            "始値",
            "高値",
            "安値",
            "前日終値",
            "出来高",
            "予想PER",
            "前日終値_MA5乖離率",
            "前日終値_MA25乖離率",
            "MA5向き",
            "MA25向き",
            "BB位置",
            "前日終値_MA5判定",
            "前日終値_MA25判定",
            "前日陽線陰線",
            "前日騰落率",
            "値幅率",
            "実体率",
            "上ヒゲ率",
            "下ヒゲ率",
            "出来高倍率25",
        ]

        for col in stock_shift_cols:
            if col in df.columns:
                df[f"予測用_{col}"] = df.groupby("Code")[col].shift(1)

        # ==========================
        # 指数：前日情報にずらす
        # ==========================
        index_cols = index_list.copy()
        
        for col in index_cols:
            if col in df.columns:
                df[f"予測用_{col}"] = df.groupby("Code")[col].shift(1)

        
        feature_df = df.copy()

        return feature_df

# ４．予測モデル
class Predictor:
    
    def __init__(self, stock_list: list = STOCK_CODES):

        self.stock_list = stock_list

    def prediction_today(self, df: pd.DataFrame, ) -> pd.DataFrame:
        """
        銘柄ごとに別モデルを学習し、
        テスト精度と最新データの上昇予測確率を返す。
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        # 使用する特徴量
        select_feaures = [
            '予測用_終値', 
            '予測用_始値', 
            '予測用_高値', 
            '予測用_安値', 
            '予測用_前日終値',

            '予測用_出来高', 
            # '予測用_予想PER', 

            '予測用_前日終値_MA5乖離率', 
            '予測用_前日終値_MA25乖離率',
            '予測用_MA5向き', 
            '予測用_MA25向き', 
            '予測用_前日終値_MA5判定', 
            '予測用_前日終値_MA25判定',

            '予測用_前日騰落率', 
            '予測用_値幅率', 
            '予測用_実体率', 
            '予測用_上ヒゲ率', 
            '予測用_下ヒゲ率',
            '予測用_出来高倍率25', 

            '予測用_^N225', 
            '予測用_^NDX', 
            '予測用_^DJI', 
            '予測用_^SPX',
            '予測用_^SOX', 
            '予測用_USDJPY=X', 
            '予測用_^VIX'
        ]

        result_list = []

        df = df.copy()
        df["Code"] = df["Code"].astype(str)
        df["日付"] = pd.to_datetime(df["日付"])

        for target_code in self.stock_list:
            target_code = str(target_code)

            # 対象銘柄だけ抽出し、時系列順に並べる
            model_df = (
                df[df["Code"] == target_code]
                .sort_values("日付")
                .copy()
            )

            if model_df.empty:
                print(f"{target_code}: データがありません")
                continue

            stock_name = model_df["銘柄名"].iloc[0]

            # 説明変数
            X_raw = model_df[select_feaures].copy()

            # カテゴリ変数をダミー化
            X_encoded = pd.get_dummies(
                X_raw,
                drop_first=True
            )

            meta_cols = model_df[
                ["日付", "銘柄名", "Code", "始値", "終値"]
            ].copy()

            y = model_df["target"].copy()

            # 同じインデックスのまま結合
            data = pd.concat(
                [meta_cols, X_encoded, y],
                axis=1
            ).dropna()

            # 学習可能な件数が少ない場合
            if len(data) < 50:
                print(
                    f"{target_code} {stock_name}: "
                    f"データ不足（{len(data)}件）"
                )
                continue

            # targetが一方のクラスしかない場合は分類不能
            if data["target"].nunique() < 2:
                print(
                    f"{target_code} {stock_name}: "
                    "targetが1種類しかありません"
                )
                continue

            # 時系列順を再確認
            data = data.sort_values("日付").reset_index(drop=True)

            split_idx = int(len(data) * 0.8)

            train_df = data.iloc[:split_idx].copy()
            test_df = data.iloc[split_idx:].copy()

            drop_cols = [
                "日付",
                "銘柄名",
                "Code",
                "target",
                "始値",
                "終値",
            ]

            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df["target"]

            X_test = test_df.drop(columns=drop_cols)
            y_test = test_df["target"]

            # 学習データに0と1の両方が存在するか確認
            if y_train.nunique() < 2:
                print(
                    f"{target_code} {stock_name}: "
                    "学習期間のtargetが1種類しかありません"
                )
                continue

            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=5,
                random_state=42,
            )

            model.fit(X_train, y_train)

            # テスト期間の評価
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # 最新行を予測
            latest_row = data.iloc[[-1]].copy()
            latest_X = latest_row.drop(columns=drop_cols)

            # 念のため学習時と列を揃える
            latest_X = latest_X.reindex(
                columns=X_train.columns,
                fill_value=0
            )

            probability = model.predict_proba(latest_X)[0, 1]
            prediction = int(probability >= 0.5)

            importance = (
                pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": model.feature_importances_
                })
                .sort_values("importance", ascending=False)
            )

            result_list.append({
                "日付": latest_row["日付"].iloc[0],
                "Code": target_code,
                "銘柄名": stock_name,
                "データ件数": len(data),
                "学習件数": len(train_df),
                "テスト件数": len(test_df),
                "テスト精度": accuracy,
                "予測確率": probability,
                "予測": prediction,
                "重要度": importance["feature"].iloc[1],
                "重要度数": importance["importance"].iloc[1]
            })

        pred_df_stock = pd.DataFrame(result_list)

        if not pred_df_stock.empty:
            pred_df_stock = (
                pred_df_stock
                .sort_values("予測確率", ascending=False)
                .reset_index(drop=True)
            )

        return pred_df_stock

# ５．通知（Discord）
class DiscordNotifier:
    """Discordへの通知を担当するクラス（単一責任の原則）"""

    def __init__(self, webhook_url: str = DISCORD_WEBHOOK_URL):
        self.webhook_url = webhook_url

    def make_discord_message(self, pred_df: pd.DataFrame) -> str:
        """予測データからDiscord用のメッセージ文字列を作成する（ロジックの分離）"""
        df = pred_df.copy()
        df["予測確率"] = (df["予測確率"] * 100).round(1)

        message = "【本日の株価予測ランキング】\n\n"
        for _, row in df.iterrows():
            message += f"{row['銘柄名']}：{row['予測確率']}%\n"

        return message

    def send_discord(self, pred_df: pd.DataFrame) -> bool:
        """Discordにメッセージを送信する（外部通信とエラーハンドリング）"""
        message = self.make_discord_message(pred_df)
        payload = {"content": message}

        try:
            # timeoutを設定して、Discord側が重いときにプログラムが無限に止まるのを防ぐ
            response = requests.post(
                self.webhook_url, json=payload, timeout=10
            )

            # ステータスコードが200番台でない場合に例外（HTTPError）を発生させる
            response.raise_for_status()

            logger.info(
                f"Discord通知に成功しました。ステータスコード: {response.status_code}"
            )
            return True

        except requests.exceptions.RequestException as e:
            # ネットワークエラーやWebhook URLの間違いなどが発生した場合、ログに記録して安全に処理を続ける
            logger.error(f"Discord通知に失敗しました: {e}")
            return False


    

if __name__ == "__main__":

    # ----- 1.データ取得 --------------------------------------------------------
    data_acquisition = DataAcquisition(stock_list, index_list)

    # 各メソッドの引数に保存先CSVパスを渡し、取得・更新・結合を自動化
    stock_df = data_acquisition.fetch_stock_data("stock_data.csv")
    index_df = data_acquisition.fetch_index_data("index_data.csv")

    # ----- 2.前処理 --------------------------------------------------------
    # テスト用データ読み込み
    stock_df = pd.read_csv("stock_data.csv")
    index_df = pd.read_csv("index_data.csv")

    prprocessor = DataPreprocessor(stock_df, index_df)
    clean_df = prprocessor.process()

    # ----- 3.特徴量生成 -----------------------------------------------------
    feature = FeatureEngineer(INDEX_CODES)
    feature_df = feature.create_features(clean_df)

    # ----- 4.予測 -----------------------------------------------------
    predictor = Predictor(STOCK_CODES)
    pred_df = predictor.prediction_today(feature_df)

    print("\n 今日の予測結果↓↓")
    print(pred_df)

    notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
    notifier.send_discord(pred_df)

    print("\n Discordに通知しました")
    
    print("\n テストおわり\n")