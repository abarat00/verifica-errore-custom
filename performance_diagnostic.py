import os
import argparse
import numpy as np
import pandas as pd
import torch

from portfolio_agent import PortfolioAgent
from portfolio_env import PortfolioEnvironment
from portfolio_models import EnhancedPortfolioActor

BASE_PATH = r'C:\Users\Administrator\Desktop\DRL PORTFOLIO\NAS Results\Multi_Ticker\Normalized_RL_INPUT'
NORM_PARAMS_PATH_BASE = os.path.join(BASE_PATH, "json")
CSV_PATH_BASE = BASE_PATH

norm_columns = [
    "open","volume","change","day","week","adjCloseGold","adjCloseSpy",
    "Credit_Spread","m_plus","m_minus","drawdown","drawup","s_plus","s_minus",
    "upper_bound","lower_bound","avg_duration","avg_depth","cdar_95","VIX_Close",
    "MACD","MACD_Signal","MACD_Histogram","SMA5","SMA10","SMA15","SMA20","SMA25",
    "SMA30","SMA36","RSI5","RSI14","RSI20","RSI25","ADX5","ADX10","ADX15","ADX20",
    "ADX25","ADX30","ADX35","BollingerLower","BollingerUpper","WR5","WR14","WR20",
    "WR25","SMA5_SMA20","SMA5_SMA36","SMA20_SMA36","SMA5_Above_SMA20","Golden_Cross",
    "Death_Cross","BB_Position","BB_Width","BB_Upper_Distance","BB_Lower_Distance",
    "Volume_SMA20","Volume_Change_Pct","Volume_1d_Change_Pct","Volume_Spike",
    "Volume_Collapse","GARCH_Vol","pred_lstm","pred_gru","pred_blstm",
    "pred_lstm_direction","pred_gru_direction","pred_blstm_direction"
]
features_per_asset = len(norm_columns)

def load_data_for_tickers(tickers, train_fraction=0.8):
    dfs_train, dfs_test, norm_params_paths, valid = {}, {}, {}, []
    for t in tickers:
        csv = os.path.join(CSV_PATH_BASE, t, f"{t}_normalized.csv")
        jsonp = os.path.join(NORM_PARAMS_PATH_BASE, f"{t}_norm_params.json")
        if os.path.exists(csv) and os.path.exists(jsonp):
            df = pd.read_csv(csv, parse_dates=['date']).sort_values('date').reset_index(drop=True)
            split = int(len(df)*train_fraction)
            dfs_train[t], dfs_test[t] = df.iloc[:split], df.iloc[split:]
            norm_params_paths[t] = jsonp
            valid.append(t)
    return dfs_train, dfs_test, norm_params_paths, valid

def align_dataframes(dfs):
    start = max(df['date'].min() for df in dfs.values())
    end = min(df['date'].max() for df in dfs.values())
    aligned = {t: df[(df['date']>=start)&(df['date']<=end)].reset_index(drop=True) for t, df in dfs.items()}
    min_len = min(len(df) for df in aligned.values())
    return {t: df.iloc[:min_len] for t, df in aligned.items()}

def load_filtered_state(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    filtered = {k: v for k, v in ckpt.items() if k in model_dict and v.size() == model_dict[k].size()}
    model.load_state_dict(filtered, strict=False)

def main(args):
    tickers = ["ARKG","IBB","IHI","IYH","XBI","VHT"]
    _, dfs_test, norm_params_paths, valid = load_data_for_tickers(tickers)
    aligned = align_dataframes(dfs_test)

    env = PortfolioEnvironment(
        tickers=valid,
        dfs=aligned,
        norm_params_paths=norm_params_paths,
        norm_columns=norm_columns,
        max_step=len(next(iter(aligned.values())))
    )

    agent = PortfolioAgent(
        num_assets=len(valid),
        memory_type="prioritized",
        batch_size=256,
        max_step=env.max_step,
        theta=0.1,
        sigma=0.2,
        use_enhanced_actor=True,
        use_batch_norm=True
    )

    agent.actor_local = EnhancedPortfolioActor(env.state_size, len(valid), features_per_asset).to(agent.device)
    agent.actor_target = EnhancedPortfolioActor(env.state_size, len(valid), features_per_asset).to(agent.device)

    load_filtered_state(agent.actor_local, args.model_path, agent.device)
    load_filtered_state(agent.actor_target, args.model_path, agent.device)

    env.reset()
    state = env.get_state()
    while not env.done:
        action = agent.act(state, noise=False)
        env.step(action)
        state = env.get_state()

    metrics = env.get_real_portfolio_metrics()
    print("\n=== PERFORMANCE DIAGNOSTIC ===")
    for key, val in metrics.items():
        print(f"{key.replace('_',' ').title()}: {val:.2f}")
    print(f"Number of Trades: {sum(np.any(np.abs(a)>1e-6) for a in env.action_history)}")
    print(f"Final Positions: {env.positions}")
    print(f"Cash Start → End: {env.cash_history[0]:.2f} → {env.cash_history[-1]:.2f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Diagnostica performance portafoglio")
    parser.add_argument("--model_path", required=True, help="Percorso al file .pth dell'actor")
    args = parser.parse_args()
    main(args)