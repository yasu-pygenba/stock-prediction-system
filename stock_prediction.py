import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta

import logging
import requests
from config import DISCORD_WEBHOOK_URL
from config import STOCK_CODES, INDEX_CODES, STOCK_NAMES

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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
                # new_df['Date'] = pd.to_datetime(new_df['Date'], utc=True).dt.tz_localize(None)
                new_df['Date'] = pd.to_datetime(new_df['Date']).dt.tz_localize(None).dt.normalize()
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                combined_df = old_df
        else:
            if not new_df.empty:
                # new_df["Date"] = pd.to_datetime(new_df["Date"], utc=True).dt.tz_localize(None)
                new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.tz_localize(None).dt.normalize()
            combined_df = new_df

        if not combined_df.empty:
            # combined_df["Date"] = pd.to_datetime(combined_df["Date"], utc=True).dt.tz_localize(None)
            combined_df["Date"] = pd.to_datetime(combined_df["Date"]).dt.tz_localize(None).dt.normalize()
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
                new_df['Date'] = pd.to_datetime(new_df['Date'].astype(str).str[:10])
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                combined_df = old_df
        else:
            if not new_df.empty:
                new_df["Date"] = pd.to_datetime(new_df["Date"].astype(str).str[:10])
            combined_df = new_df
        
        if not combined_df.empty:
            combined_df["Date"] = pd.to_datetime(combined_df["Date"].astype(str).str[:10])
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
            "出来高",
            "予想PER",
            "前日終値_MA5乖離率",
            "前日終値_MA25乖離率",
            "MA5向き",
            "MA25向き",
            "BB位置",
            "前日終値_MA5判定",
            "前日終値_MA25判定",
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

    def prediction_today(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        銘柄ごとに別モデルを学習し、重要度トップ10の特徴量を再選択して再学習。
        テスト精度と最新データの上昇予測確率を返す。
        """

        # 使用する特徴量
        select_features = [
            '予測用_終値', '予測用_始値', '予測用_高値', '予測用_安値', '予測用_出来高', 
            '予測用_前日終値_MA5乖離率', '予測用_前日終値_MA25乖離率',
            '予測用_MA5向き', '予測用_MA25向き', '予測用_前日終値_MA5判定', '予測用_前日終値_MA25判定',
            # '前日騰落率', 
            '予測用_値幅率', '予測用_実体率', '予測用_上ヒゲ率', '予測用_下ヒゲ率',
            '予測用_出来高倍率25', 
            '予測用_^N225', '予測用_NIY=F', '予測用_^NDX', '予測用_^DJI', '予測用_^SPX',
            '予測用_^SOX', '予測用_USDJPY=X', '予測用_^VIX'
        ]

        result_list = []
        backtest_list = []

        df = df.copy()
        df["Code"] = df["Code"].astype(str)
        df["日付"] = pd.to_datetime(df["日付"])

        for target_code in self.stock_list:
            target_code = str(target_code)

            # 対象銘柄だけ抽出し、時系列順に並べる
            model_df = df[df["Code"] == target_code].sort_values("日付").copy()

            if model_df.empty:
                print(f"{target_code}: データがありません")
                continue

            stock_name = model_df["銘柄名"].iloc[0]

            # 説明変数
            X_raw = model_df[select_features].copy()

            # カテゴリ変数をダミー化
            X_encoded = pd.get_dummies(X_raw, drop_first=True)

            meta_cols = model_df[["日付", "銘柄名", "Code", "始値", "終値"]].copy()
            y = model_df["target"].copy()

            # 同じインデックスのまま結合
            # data = pd.concat([meta_cols, X_encoded, y], axis=1).dropna()

            data_all = pd.concat([meta_cols, X_encoded, y], axis=1)

            data_all = data_all.sort_values("日付").reset_index(drop=True)

            latest_row = data_all.iloc[[-1]].copy()

            historical_data = data_all.iloc[:-1].copy()
            data = historical_data.dropna()

            if len(data) < 50:
                print(f"{target_code} {stock_name}: データ不足（{len(data)}件）")
                continue

            if data["target"].nunique() < 2:
                print(f"{target_code} {stock_name}: targetが1種類しかありません")
                continue

            # 時系列順を再確認
            data = data.sort_values("日付").reset_index(drop=True)

            split_idx = int(len(data) * 0.8)
            train_df = data.iloc[:split_idx].copy()
            test_df = data.iloc[split_idx:].copy()

            drop_cols = ["日付", "銘柄名", "Code", "target", "始値", "終値"]

            X_train_full = train_df.drop(columns=drop_cols)
            y_train = train_df["target"]
            X_test_full = test_df.drop(columns=drop_cols)
            y_test = test_df["target"]

            if y_train.nunique() < 2:
                print(f"{target_code} {stock_name}: 学習期間のtargetが1種類しかありません")
                continue

            # --- 1回目の学習（特徴量の重要度を測定するため） ---
            initial_model = RandomForestClassifier(
                n_estimators=300,  # 1回目は特徴量選定用なので少し少なめでもOK
                max_depth=4,
                min_samples_leaf=5,
                random_state=42,
            )
            initial_model.fit(X_train_full, y_train)

            # 重要度の算出と上位10個の抽出
            importance_df = pd.DataFrame({
                "feature": X_train_full.columns,
                "importance": initial_model.feature_importances_
            }).sort_values("importance", ascending=False)
            
            # 上位10個の特徴量名を取得
            top_n = 10
            top_features = importance_df["feature"].head(top_n).tolist()

            # --- 2回目の学習（選ばれたトップ10の特徴量だけで再学習） ---
            X_train_selected = X_train_full[top_features].copy()
            X_test_selected = X_test_full[top_features].copy()

            final_model = RandomForestClassifier(
                n_estimators=300,  # 本番用モデルなので300本でじっくり学習
                max_depth=4,
                min_samples_leaf=5,
                random_state=42,
            )
            final_model.fit(X_train_selected, y_train)

            # バックテスト用リストに保存
            test_proba = final_model.predict_proba(X_test_selected)[:, 1]
            y_pred = (test_proba >= 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)

            test_results = test_df[["日付", "Code", "銘柄名"]].copy()
            test_results["予測確率"] = test_proba
            backtest_list.append(test_results)

            # 最新行を予測
            latest_X_full = latest_row.drop(columns=drop_cols)
            
            # 学習時と列を揃えつつ、トップ10のみに絞り込む
            latest_X_selected = latest_X_full.reindex(columns=X_train_full.columns, fill_value=0)
            latest_X_selected = latest_X_selected[top_features]

            latest_X_selected = latest_X_selected.fillna(0)

            probability = final_model.predict_proba(latest_X_selected)[0, 1]
            prediction = int(probability >= 0.5)

            # ログ用：この銘柄で一番重要だった特徴量（1位と2位）
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
                "重要度トップ1": importance_df["feature"].iloc[0],
                "重要度数トップ1": importance_df["importance"].iloc[0],
                "重要度トップ2": importance_df["feature"].iloc[1],
                "重要度数トップ2": importance_df["importance"].iloc[1]
            })

        pred_df_stock = pd.DataFrame(result_list)

        if not pred_df_stock.empty:
            pred_df_stock = (
                pred_df_stock
                .sort_values("予測確率", ascending=False)
                .reset_index(drop=True)
            )

        if backtest_list:
            backtest_pred_df = pd.concat(backtest_list, ignore_index=True)
        else:
            backtest_pred_df = pd.DataFrame()

        return pred_df_stock, backtest_pred_df

# ５．通知（Discord）
class DiscordNotifier:
    """Discordへの通知を担当するクラス"""

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

# ６．バックテスト
class BackTester:
    """
    予測モデルの結果に基づいたバックテストを実行
    ロング（買い）およびショート（空売り）戦略のパフォーマンスを集計するクラス
    予測結果=>0.5 :
    　１．寄り買いして引け売りした時の損益
    　２．寄り買いして利確ライン5%、損切ライン3%にした時の損益
    予測結果<0.5 :
    　１．寄り空売りして引け買い戻した時の損益
    """

    def __init__(self, take_profit: float = 1.05, stop_loss: float = 0.97):
        """
        Args:
            take_profit:利確ライン（例:1.05 = 5%）
            stop_loss:損切ライン（例：0.97 = -3%)
        """
        self.take_profit = take_profit
        self.stop_loss = stop_loss

    def _calculate_oco_profit(self, row: pd.Series) -> dict:
        """
        1日の中での利確・損切（OCO）のシミュレーション（買い限定）
        """
        entry = row["始値"]
        # 利確・損切価格を算出（100円単位などに丸めず実数で計算）
        tp_price = entry * self.take_profit
        sl_price = entry * self.stop_loss

        high = row["高値"]
        low = row["安値"]
        close = row["終値"]

        # 損切優先のロジック
        if low <= sl_price:
            exit_price = sl_price
            result = "損切"
        elif high >= tp_price:
            exit_price = tp_price
            result = "利確"
        else:
            exit_price = close
            result = "引け"
        
        profit_yen = exit_price - entry
        profit_rate = (profit_yen / entry) * 100

        return {"損益額_1株": profit_yen, "損益率": profit_rate, "結果": result}

    def run_backtest(self, backtest_pred_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        予測結果と株価データを結合し、日ごとのバックテスト結果を算出する

        Args:
            prediction_results_df: 日付, Code, 予想確率 が入ったDataFrame
            price_df: 日付, Code, 始値, 高値, 安値, 終値 が入ったDataFrame
        """
        # データのコピーと型変換
        pred_df = backtest_pred_df.copy()
        prices = price_df.copy()

        pred_df["Code"] = pred_df["Code"].astype(str)
        prices["Code"] = prices["Code"].astype(str)
        pred_df["日付"] = pd.to_datetime(pred_df["日付"])
        prices["日付"] = pd.to_datetime(prices["日付"])

        # 予測データと株価データを日付とCodeをキーに結合
        bt_data = pd.merge(pred_df, prices, on=["日付", "Code"], how="inner")

        reults = []

        for _, row in bt_data.iterrows():
            prob = row["予測確率"]
            entry = row["始値"]
            close = row["終値"]

            trade_info = {
                "日付": row["日付"],
                "Code": row["Code"],
                "銘柄名": row.get("銘柄名", ""),
                "予測確率": prob,
                "始値": entry,
                "終値": close,
                "戦略": "待機",
                "損益額_1株": 0.0,
                "損益率": 0.0,
                "OCO結果": "対象外",
            }
        
            # ① 予想確率 0.5 以上　→　買い（ロングエントリー）
            if prob >= 0.5:
                # 寄り買い・引け売り
                profit_yen = close - entry
                profit_rate = (profit_yen / entry) * 100

                # OCO（利確・損切）の計算
                oco_res = self._calculate_oco_profit(row)

                trade_info.update({
                    "戦略": "寄り買い_引け売り",
                    "損益額_1株": profit_yen,
                    "損益率": profit_rate,
                    "OCO_損益額_1株": oco_res["損益額_1株"],
                    "OCO_損益率": oco_res["損益率"],
                    "OCO結果": oco_res["結果"]
                })

            # ② 予想確率 0.5 未満　→　空売り（ショートエントリー）
            else:
                # 寄り空売り・引け買い戻し
                profit_yen = - (close - entry)
                profit_rate = (profit_yen / entry) * 100

                trade_info.update({
                    "戦略": "寄り空売り_引け買い戻し",
                    "損益額_1株": profit_yen,
                    "損益率": profit_rate,
                    # 空売りの日中OCOはまだ未実装
                    "OCO_損益額_1株": profit_yen,
                    "OCO_損益率": profit_rate,
                    "OCO結果": "引け（空売り）"
                })
            
            reults.append(trade_info)

        return pd.DataFrame(reults)
    
    def summarize_performance(self, detailed_result_df: pd.DataFrame) -> pd.DataFrame:
        """
        詳細なバックテスト結果から、戦略ごとの累計損益や勝率をまとめる
        """
        df = detailed_result_df.copy()
        summary_list = []

        for strategy_name in ["寄り買い_引け売り", "寄り空売り_引け買い戻し"]:
            strat_df = df[df["戦略"] == strategy_name]

            if strat_df.empty:
                continue

            total_profit_rate = strat_df["損益率"].sum()
            total_profit_yen = strat_df["損益額_1株"].sum()
            win_rate = strat_df["損益額_1株"].gt(0).mean() * 100
            trade_count = len(strat_df)

            summary_list.append({
                "戦略": strategy_name,
                "トレード回数": trade_count,
                "累計損益率(%)": round(total_profit_rate, 2),
                "累計損益額(円)": total_profit_yen,
                "勝率(%)": round(win_rate, 2)
            })
        
        # 買い戦略における OCO（利確・損切）パターンの集計も追加
        buy_strat_df = df[df["戦略"] == "寄り買い_引け売り"]
        if not buy_strat_df.empty:
            summary_list.append({
                "戦略": f"寄り買い_OCO（利確:{self.take_profit}, 損切:{self.stop_loss})",
                "トレード回数": len(buy_strat_df),
                "累計損益率(%)": round(buy_strat_df["OCO_損益率"].sum(), 2),
                "累計損益額(円)": buy_strat_df["OCO_損益額_1株"].sum(),
                "勝率(%)": round(buy_strat_df["OCO_損益額_1株"].gt(0).mean() * 100, 2),
            })

        return pd.DataFrame(summary_list)
    

if __name__ == "__main__":

    # ----- 1.データ取得 --------------------------------------------------------
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

    # ----- 3.特徴量生成 -----------------------------------------------------
    feature = FeatureEngineer(INDEX_CODES)
    feature_df = feature.create_features(clean_df)

    # ----- 4.予測 ---------------------------------------------------------
    predictor = Predictor(STOCK_CODES)
    pred_df, backtest_pred_df = predictor.prediction_today(feature_df)

    print("\n 今日の予測結果↓↓ \n")
    print(pred_df.head(3))

    # ----- 5.通知 ---------------------------------------------------------
    # notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
    # notifier.send_discord(pred_df)

    # print("\n Discordに通知しました")

    # ----- 6.バックテスト ---------------------------------------------------
    backtester = BackTester(take_profit=1.05, stop_loss=0.97)
    bt_df = backtester.run_backtest(backtest_pred_df=backtest_pred_df, price_df=clean_df)

    print("\n バックテストが終わりました \n")
    if not bt_df.empty:
        latest_date = pred_df["日付"].max()
        latest_bt_df = pred_df[pred_df["日付"] == latest_date]

        # 日付の表記を YY-MM-DD に整えて表示
        print(f"\n【直近取引日（{latest_date.strftime('%Y-%m-%d')}）のバックテスト結果】")
        print(latest_bt_df.drop(columns=["日付"]).sort_values("予測確率", ascending=False))
    else:
        print("\n バックテストのデータがありませんでした。")
        
    summary_df = backtester.summarize_performance(bt_df)

    print("\n 【全期間の集計結果】 \n")
    print(summary_df)
    
    print("\n テストおわり\n")