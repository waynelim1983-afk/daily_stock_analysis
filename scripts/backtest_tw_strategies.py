#!/usr/bin/env python3
"""
台股三策略量化回測腳本
策略A：均線金叉 + 放量突破
策略B：多頭回踩 + 縮量
策略C：VCP（波動收縮形態）
"""

import argparse
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# 台股清單
# ---------------------------------------------------------------------------

FALLBACK_TW_STOCKS = [
    "2330", "2317", "2454", "2412", "2308", "2382", "2881", "2882", "2886", "2884",
    "2891", "2892", "2885", "2883", "2887", "1301", "1303", "1326", "2002", "2006",
    "2207", "2301", "2303", "2357", "2379", "2395", "2408", "2409", "2474", "2498",
    "2603", "2609", "2615", "2618", "2633", "2801", "2823", "2880", "2888", "2890",
    "3008", "3034", "3037", "3045", "3481", "3673", "3711", "4904", "4938", "6505",
    "6669", "5871", "6415", "8046", "9910", "2105", "2201", "2049", "1590", "3006",
]


def get_tw_stock_list() -> list[str]:
    """取得台股上市股票代號清單（優先用 akshare，失敗改用備用清單）。"""
    try:
        import akshare as ak  # type: ignore
        df = ak.stock_info_tw_name_code(indicator="上市")
        codes = df.iloc[:, 0].astype(str).tolist()
        codes = [c.strip() for c in codes if c.strip().isdigit()]
        if len(codes) >= 50:
            print(f"[akshare] 取得台股上市清單：{len(codes)} 支")
            return codes
        print(f"[akshare] 清單不足（{len(codes)} 支），改用備用清單")
    except Exception as e:
        print(f"[akshare] 取得清單失敗：{e}，改用備用清單")
    return list(FALLBACK_TW_STOCKS)


# ---------------------------------------------------------------------------
# 資料下載
# ---------------------------------------------------------------------------

def download_history(code: str, start: str, end: str) -> pd.DataFrame | None:
    """下載 yfinance 日線資料，失敗回傳 None。"""
    ticker = f"{code}.TW"
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty or len(df) < 60:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 技術指標計算
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算策略所需的所有技術指標。"""
    d = df.copy()
    d["ma5"]   = d["close"].rolling(5).mean()
    d["ma10"]  = d["close"].rolling(10).mean()
    d["ma20"]  = d["close"].rolling(20).mean()
    d["ma60"]  = d["close"].rolling(60).mean()
    d["ma200"] = d["close"].rolling(200).mean()

    d["vol_ma5"]  = d["volume"].rolling(5).mean()
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    return d


def avg_daily_volume_shares(df: pd.DataFrame) -> float:
    """計算最近 20 日平均成交量（股數）。"""
    return df["volume"].iloc[-20:].mean() if len(df) >= 20 else 0.0


# ---------------------------------------------------------------------------
# 策略訊號函數
# ---------------------------------------------------------------------------

def signal_a_ma_golden_cross(df: pd.DataFrame) -> pd.Series:
    """
    策略A：均線金叉 + 放量突破
    - MA5 今日上穿 MA10
    - 收盤突破近 60 日最高點（不含今日）
    - 成交量 > 20日均量 × 1.5
    """
    d = df.copy()
    d["high60_prev"] = d["close"].shift(1).rolling(60).max()
    cond_cross   = (d["ma5"] > d["ma10"]) & (d["ma5"].shift(1) <= d["ma10"].shift(1))
    cond_break   = d["close"] > d["high60_prev"]
    cond_volume  = d["volume"] > d["vol_ma20"] * 1.5
    return (cond_cross & cond_break & cond_volume).fillna(False)


def signal_b_bull_pullback(df: pd.DataFrame) -> pd.Series:
    """
    策略B：多頭回踩 + 縮量
    - MA5 > MA10 > MA20（多頭排列）
    - 今日收盤 < 昨日收盤（回踩）
    - 成交量 < 5日均量 × 0.8（縮量）
    - 收盤仍 > MA20（未破主要支撐）
    """
    d = df.copy()
    cond_trend   = (d["ma5"] > d["ma10"]) & (d["ma10"] > d["ma20"])
    cond_pullback = d["close"] < d["close"].shift(1)
    cond_shrink  = d["volume"] < d["vol_ma5"] * 0.8
    cond_support = d["close"] > d["ma20"]
    return (cond_trend & cond_pullback & cond_shrink & cond_support).fillna(False)


def _vcp_contraction_check(df: pd.DataFrame, idx: int) -> bool:
    """
    檢查 idx 位置是否滿足 VCP 震幅收縮條件（往前 60 根）。
    把 60 日等分成 3 段，每段震幅比前段小至少 25%，最後段震幅 < 10%。
    """
    start = idx - 59
    if start < 0:
        return False
    segment_len = 20  # 60 / 3
    amplitudes = []
    for seg in range(3):
        s = start + seg * segment_len
        e = s + segment_len
        seg_data = df.iloc[s:e]
        if seg_data.empty:
            return False
        mid_close = seg_data["close"].median()
        if mid_close <= 0:
            return False
        amp = (seg_data["high"].max() - seg_data["low"].min()) / mid_close
        amplitudes.append(amp)

    # 每段震幅需比前段小至少 25%
    for i in range(1, len(amplitudes)):
        if amplitudes[i] >= amplitudes[i - 1] * 0.75:
            return False

    # 最後一段震幅需 < 10%
    if amplitudes[-1] >= 0.10:
        return False

    return True


def signal_c_vcp(df: pd.DataFrame) -> pd.Series:
    """
    策略C：VCP（波動收縮形態）
    - 近 60 日分 3 段，每段震幅比前段小至少 25%
    - 最後一段震幅 < 10%
    - 收盤在 MA200 之上
    - 成交量 > 20日均量 × 1.5
    - 收盤突破近 60 日最高點
    """
    d = df.copy()
    d["high60_prev"] = d["close"].shift(1).rolling(60).max()
    cond_ma200   = d["close"] > d["ma200"]
    cond_volume  = d["volume"] > d["vol_ma20"] * 1.5
    cond_break   = d["close"] > d["high60_prev"]

    signals = pd.Series(False, index=d.index)
    valid_mask = cond_ma200 & cond_volume & cond_break

    for i, (idx, row) in enumerate(d.iterrows()):
        if not valid_mask.iloc[i]:
            continue
        if _vcp_contraction_check(d, i):
            signals.iloc[i] = True

    return signals


# ---------------------------------------------------------------------------
# 回測模擬
# ---------------------------------------------------------------------------

HOLD_DAYS = 20
STOP_LOSS = -0.07  # -7%


def simulate_trades(df: pd.DataFrame, signals: pd.Series) -> list[dict]:
    """
    模擬交易：進場=訊號日隔日開盤，出場=持有20個交易日或觸及 -7% 停損。
    回傳每筆交易明細列表。
    """
    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    hold_count = 0

    for i in range(len(df)):
        if in_position:
            # 每日盤中最低點觸及停損
            low_return = (df["low"].iloc[i] - entry_price) / entry_price
            if low_return <= STOP_LOSS:
                # 以停損價出場（entry_price * (1 + STOP_LOSS)）
                exit_price = entry_price * (1 + STOP_LOSS)
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": df.index[i],
                    "pnl": pnl,
                    "stop_loss": True,
                })
                in_position = False
                continue

            hold_count += 1
            if hold_count >= HOLD_DAYS:
                exit_price = df["open"].iloc[i]
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": df.index[i],
                    "pnl": pnl,
                    "stop_loss": False,
                })
                in_position = False
                continue

        else:
            # 訊號日隔日開盤進場
            if i > 0 and signals.iloc[i - 1]:
                entry_price = df["open"].iloc[i]
                if entry_price <= 0:
                    continue
                entry_date = df.index[i]
                in_position = True
                hold_count = 0

    return trades


# ---------------------------------------------------------------------------
# 績效指標計算
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[dict]) -> dict:
    """計算勝率、平均損益、最大回撤、夏普比率。"""
    if not trades:
        return {
            "count": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_pnl  = float(np.mean(pnls)) if pnls else 0.0
    avg_win  = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    # 最大回撤：用累積損益序列計算
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    # 夏普比率：(年化平均報酬 - 0.02) / 年化標準差
    # 年化因子 = 252 / 20（每次持倉 20 日）
    annual_factor = 252 / HOLD_DAYS
    annual_return = avg_pnl * annual_factor
    std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0.0
    annual_std = std_pnl * (annual_factor ** 0.5)
    sharpe = (annual_return - 0.02) / annual_std if annual_std > 0 else 0.0

    return {
        "count": len(pnls),
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# 主回測流程
# ---------------------------------------------------------------------------

STRATEGIES = {
    "A": ("均線金叉 + 放量突破", signal_a_ma_golden_cross),
    "B": ("多頭回踩 + 縮量",      signal_b_bull_pullback),
    "C": ("VCP（波動收縮形態）",  signal_c_vcp),
}


def run_backtest(
    strategy_keys: list[str],
    min_volume: int,
    start_date: str,
    end_date: str,
) -> dict[str, list[dict]]:
    """對所有台股執行回測，回傳各策略的交易清單。"""
    codes = get_tw_stock_list()
    all_trades: dict[str, list[dict]] = {k: [] for k in strategy_keys}

    total = len(codes)
    valid_count = 0
    skipped_count = 0

    print(f"\n開始下載資料並回測，共 {total} 支股票...\n")

    for idx, code in enumerate(codes, 1):
        if idx % 10 == 0 or idx == total:
            print(f"  進度：{idx}/{total}（有效：{valid_count}，跳過：{skipped_count}）")

        df = download_history(code, start_date, end_date)
        if df is None:
            skipped_count += 1
            continue

        # 過濾低流動性（20日均量 < min_volume 股）
        avg_vol = avg_daily_volume_shares(df)
        if avg_vol < min_volume:
            skipped_count += 1
            continue

        df = compute_indicators(df)
        valid_count += 1

        for key in strategy_keys:
            _, signal_fn = STRATEGIES[key]
            try:
                signals = signal_fn(df)
                trades = simulate_trades(df, signals)
                all_trades[key].extend(trades)
            except Exception as e:
                # 單一策略/股票錯誤不中斷整體
                pass

    print(f"\n篩選完成：有效股票 {valid_count} 支，跳過 {skipped_count} 支\n")
    return all_trades, valid_count


# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------

def print_results(
    all_trades: dict[str, list[dict]],
    strategy_keys: list[str],
    valid_count: int,
    start_date: str,
    end_date: str,
) -> None:
    """輸出繁體中文回測結果。"""
    SEP = "=" * 60

    print(SEP)
    print("📊 台股策略回測結果")
    print(f"回測期間：{start_date} ~ {end_date}")
    print(f"有效股票數：{valid_count} 支（過濾後）")
    print(SEP)

    for key in strategy_keys:
        name, _ = STRATEGIES[key]
        trades = all_trades[key]
        m = compute_metrics(trades)

        print(f"\n📈 策略{key}：{name}")
        print(f"  總交易次數：{m['count']}")
        if m["count"] == 0:
            print("  （無交易紀錄）")
            continue
        sign_avg  = "+" if m["avg_pnl"]  >= 0 else ""
        sign_win  = "+" if m["avg_win"]  >= 0 else ""
        sign_loss = "+" if m["avg_loss"] >= 0 else ""
        print(f"  勝率：{m['win_rate'] * 100:.1f}%")
        print(f"  平均損益：{sign_avg}{m['avg_pnl'] * 100:.2f}%")
        print(f"  平均獲利：{sign_win}{m['avg_win'] * 100:.2f}%")
        print(f"  平均虧損：{sign_loss}{m['avg_loss'] * 100:.2f}%")
        print(f"  最大回撤：{m['max_drawdown'] * 100:.2f}%")
        print(f"  夏普比率：{m['sharpe']:.2f}")

    print(f"\n{SEP}")
    print("💡 說明")
    print("- 進場：訊號日隔日開盤")
    print(f"- 出場：持有{HOLD_DAYS}個交易日或觸及-7%停損")
    print("- 成交量篩選：20日均量 < 500張 的股票已排除")
    print(SEP)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="台股三策略量化回測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
策略說明：
  A  均線金叉 + 放量突破
  B  多頭回踩 + 縮量
  C  VCP（波動收縮形態）
        """,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="A,B,C",
        help="指定回測策略，多個以逗號分隔，例如 A,B 或 C（預設：A,B,C）",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=500_000,
        help="最低 20 日均量（股數，預設：500000，即 500 張）",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        metavar="FILE",
        help="將回測結果摘要寫入 JSON 檔案（供 CI/通知腳本讀取）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 解析策略清單
    raw_keys = [k.strip().upper() for k in args.strategy.split(",")]
    valid_keys = [k for k in raw_keys if k in STRATEGIES]
    invalid_keys = [k for k in raw_keys if k not in STRATEGIES]
    if invalid_keys:
        print(f"[警告] 未知策略：{invalid_keys}，已忽略")
    if not valid_keys:
        print("錯誤：沒有有效的策略，請指定 A、B 或 C")
        sys.exit(1)

    # 回測期間：近 3 年
    end_date   = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    print(f"策略：{', '.join(valid_keys)}")
    print(f"最低成交量篩選：{args.min_volume:,} 股（{args.min_volume // 1000} 張）")
    print(f"回測期間：{start_date} ~ {end_date}")

    all_trades, valid_count = run_backtest(
        strategy_keys=valid_keys,
        min_volume=args.min_volume,
        start_date=start_date,
        end_date=end_date,
    )

    print_results(
        all_trades=all_trades,
        strategy_keys=valid_keys,
        valid_count=valid_count,
        start_date=start_date,
        end_date=end_date,
    )

    if args.json_out:
        import json as _json
        summary = {
            "start_date": start_date,
            "end_date": end_date,
            "valid_count": valid_count,
            "strategies": {},
        }
        for key in valid_keys:
            name, _ = STRATEGIES[key]
            m = compute_metrics(all_trades[key])
            m["name"] = name
            summary["strategies"][key] = m
        with open(args.json_out, "w", encoding="utf-8") as _f:
            _json.dump(summary, _f, ensure_ascii=False, indent=2)
        print(f"\n回測摘要已寫入：{args.json_out}")
