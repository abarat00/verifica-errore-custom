"""
Performance Analysis for DRL Portfolio Training (No Commission)

Usage:
    python performance_analysis.py --model_path "path/to/model_checkpoint.pth" [--test_data_dir "path/to/test_data_folder"] [--output_dir "path/to/output_folder"]

Description:
    Questo script carica un modello checkpoint salvato durante l'addestramento (per esempio, un file 'portfolio_actor_XX.pth')
    e lo utilizza per eseguire una valutazione sull'ambiente di test. Vengono calcolate diverse metriche di performance,
    come il rendimento totale, Sharpe ratio, max drawdown e il valore finale del portafoglio.
    
    Le visualizzazioni (grafici) vengono salvate nella cartella di output specificata.
    
    Opzioni aggiuntive possono essere passate per indicare la directory contenente i dati di test (se diverso dal default)
    e la directory dove salvare i report.
    
Arguments:
    --model_path       Path al checkpoint del modello da analizzare.
    --test_data_dir    (Opzionale) Directory contenente i dati di test (CSV e file di normalizzazione).
    --output_dir       (Opzionale) Directory in cui salvare i risultati e i grafici di analisi.
"""

import os
import argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# Import dei moduli locali: assicurarsi che il PYTHONPATH includa la directory dei moduli
from portfolio_agent import PortfolioAgent
from portfolio_env import PortfolioEnvironment

def load_test_data(tickers, norm_params_path_base, csv_path_base, train_fraction=0.8):
    """
    Carica e allinea i dati di test per i ticker specificati.
    """
    dfs = {}
    norm_params_paths = {}
    for ticker in tickers:
        norm_params_path = os.path.join(norm_params_path_base, f"{ticker}_norm_params.json")
        csv_path = os.path.join(csv_path_base, ticker, f"{ticker}_normalized.csv")
        if not os.path.exists(norm_params_path) or not os.path.exists(csv_path):
            print(f"ATTENZIONE: File per {ticker} non trovato.")
            continue
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        # Prendiamo la parte di test
        train_size = int(len(df) * train_fraction)
        dfs[ticker] = df.iloc[train_size:].copy()
        norm_params_paths[ticker] = norm_params_path
    return dfs, norm_params_paths

def align_dataframes(dfs):
    """
    Allinea i DataFrame in modo che abbiano lo stesso intervallo di date e lo stesso numero di righe.
    """
    aligned_dfs = {}
    if all('date' in df.columns for df in dfs.values()):
        start_date = max(df['date'].min() for df in dfs.values())
        end_date = min(df['date'].max() for df in dfs.values())
        print(f"Intervallo di date comune: {start_date} - {end_date}")
        for ticker, df in dfs.items():
            aligned_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
            aligned_df = aligned_df.sort_values('date')
            aligned_dfs[ticker] = aligned_df
        lengths = [len(df) for df in aligned_dfs.values()]
        if len(set(lengths)) > 1:
            min_length = min(lengths)
            for ticker in aligned_dfs:
                aligned_dfs[ticker] = aligned_dfs[ticker].iloc[:min_length].copy()
    else:
        min_rows = min(len(df) for df in dfs.values())
        for ticker, df in dfs.items():
            aligned_dfs[ticker] = df.iloc[:min_rows].copy()
    return aligned_dfs

def plot_performance(metrics, output_dir):
    """Genera alcuni grafici delle performance e li salva nella directory di output."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Grafico del valore del portafoglio
    plt.figure(figsize=(10,6))
    plt.plot(metrics['portfolio_values'], marker='o')
    plt.title("Valore del Portafoglio nel Tempo")
    plt.xlabel("Episodi")
    plt.ylabel("Valore del Portafoglio ($)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "portfolio_value.png"))
    plt.close()

    # Grafico del rendimento cumulativo
    plt.figure(figsize=(10,6))
    plt.plot(metrics['cum_rewards'], marker='o', color='g')
    plt.title("Ricompensa Cumulativa")
    plt.xlabel("Episodi")
    plt.ylabel("Ricompensa Media")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cumulative_rewards.png"))
    plt.close()

    # Grafico dello Sharpe Ratio
    plt.figure(figsize=(10,6))
    plt.plot(metrics['sharpe_ratios'], marker='o', color='orange')
    plt.title("Sharpe Ratio nel Tempo")
    plt.xlabel("Episodi")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "sharpe_ratio.png"))
    plt.close()

def main(args):
    # Configurazione
    TICKERS = ["ARKG", "IBB", "IHI", "IYH", "XBI", "VHT"]
    BASE_PATH = 'C:\\Users\\Administrator\\Desktop\\DRL PORTFOLIO\\NAS Results\\Multi_Ticker\\Normalized_RL_INPUT\\'
    NORM_PARAMS_PATH_BASE = os.path.join(BASE_PATH, "json")
    CSV_PATH_BASE = BASE_PATH
    OUTPUT_DIR = args.output_dir if args.output_dir else "results/portfolio_analysis"
    
    # Carica i dati di test
    dfs, norm_params_paths = load_test_data(TICKERS, NORM_PARAMS_PATH_BASE, CSV_PATH_BASE)
    aligned_dfs_test = align_dataframes(dfs)
    
    # Inizializza l'ambiente di test
    from portfolio_env import PortfolioEnvironment  # Assicurarsi che il modulo sia accessibile
    # In questo esempio usiamo gli stessi parametri del run senza commissioni
    env = PortfolioEnvironment(
        tickers=TICKERS,
        lambd=0.05,
        psi=0.2,
        cost="trade_l1",
        max_pos_per_asset=2.0,
        max_portfolio_pos=6.0,
        squared_risk=False,
        penalty="tanh",
        alpha=3,
        beta=3,
        clip=True,
        scale_reward=5,
        dfs=aligned_dfs_test,
        max_step=len(next(iter(aligned_dfs_test.values()))),
        norm_params_paths=norm_params_paths,
        norm_columns=[  # La stessa lista di feature usata durante l'addestramento
            "open", "volume", "change", "day", "week", "adjCloseGold", "adjCloseSpy",
            "Credit_Spread", "m_plus", "m_minus", "drawdown", "drawup",
            "s_plus", "s_minus", "upper_bound", "lower_bound", "avg_duration", "avg_depth",
            "cdar_95", "VIX_Close", "MACD", "MACD_Signal", "MACD_Histogram", "SMA5",
            "SMA10", "SMA15", "SMA20", "SMA25", "SMA30", "SMA36", "RSI5", "RSI14", "RSI20",
            "RSI25", "ADX5", "ADX10", "ADX15", "ADX20", "ADX25", "ADX30", "ADX35",
            "BollingerLower", "BollingerUpper", "WR5", "WR14", "WR20", "WR25",
            "SMA5_SMA20", "SMA5_SMA36", "SMA20_SMA36", "SMA5_Above_SMA20",
            "Golden_Cross", "Death_Cross", "BB_Position", "BB_Width",
            "BB_Upper_Distance", "BB_Lower_Distance", "Volume_SMA20", "Volume_Change_Pct",
            "Volume_1d_Change_Pct", "Volume_Spike", "Volume_Collapse", "GARCH_Vol",
            "pred_lstm", "pred_gru", "pred_blstm", "pred_lstm_direction",
            "pred_gru_direction", "pred_blstm_direction"
        ],
        free_trades_per_month=float('inf'),
        commission_rate=0.0,
        min_commission=0.0,
        trading_frequency_penalty_factor=0.05,
        position_stability_bonus_factor=0.2,
        correlation_penalty_factor=0.15,
        diversification_bonus_factor=0.25,
        initial_capital=100000,
        risk_free_rate=0.02,
        use_sortino=True,
        target_return=0.05
    )
    
    # Inizializza l'agente per il test (non verrà addestrato qui, solo per caricare il modello)
    from portfolio_agent import PortfolioAgent  # Assicurarsi che il modulo sia accessibile
    agent = PortfolioAgent(
        num_assets=len(TICKERS),
        memory_type="prioritized",
        batch_size=256,
        max_step=env.max_step,
        theta=0.1,
        sigma=0.2,
        use_enhanced_actor=True,
        use_batch_norm=True
    )
    
    # Carica il modello specificato dall'argomento --model_path
    if args.model_path and os.path.exists(args.model_path):
        print(f"Caricamento del modello da: {args.model_path}")
        agent.load_models(actor_path=args.model_path)
    else:
        print("Errore: specificare un percorso valido per --model_path")
        return

    # Esegui la valutazione sul dataset di test
    print("Esecuzione della valutazione sul dataset di test...")
    test_rewards = []
    portfolio_values = []
    sharpe_ratios = []
    env.reset()
    state = env.get_state()
    done = env.done
    
    while not done:
        with torch.no_grad():
            actions = agent.act(state, noise=False)
        reward = env.step(actions)
        state = env.get_state()
        test_rewards.append(reward)
        metrics = env.get_real_portfolio_metrics()
        portfolio_values.append(metrics['final_portfolio_value'])
        sharpe_ratios.append(metrics['sharpe_ratio'])
        done = env.done

    avg_reward = np.mean(test_rewards)
    final_value = portfolio_values[-1] if portfolio_values else 0
    avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0

    print("\n--- Risultati della Valutazione ---")
    print(f"Ricompensa media: {avg_reward:.2f}")
    print(f"Valore finale del portafoglio: ${final_value:.2f}")
    print(f"Sharpe Ratio medio: {avg_sharpe:.2f}")

    # Salva un report in CSV
    report = pd.DataFrame({
        'episode': list(range(len(test_rewards))),
        'reward': test_rewards,
        'portfolio_value': portfolio_values,
        'sharpe_ratio': sharpe_ratios
    })
    report_file = os.path.join(args.output_dir if args.output_dir else OUTPUT_DIR, "test_report.csv")
    report.to_csv(report_file, index=False)
    print(f"Report salvato in: {report_file}")

    # Genera grafici di analisi
    plot_performance({
        'cum_rewards': np.cumsum(test_rewards),
        'portfolio_values': portfolio_values,
        'sharpe_ratios': sharpe_ratios
    }, args.output_dir if args.output_dir else OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi delle Performance per il Portafoglio RL (No Commissioni)")
    parser.add_argument("--model_path", type=str, required=True, help="Percorso al checkpoint del modello da analizzare (file .pth)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory dove salvare i report e grafici (default: results/portfolio_analysis)")
    parser.add_argument("--test_data_dir", type=str, default=None, help="(Opzionale) Directory dei dati di test, se diverso dal default")
    args = parser.parse_args()
    
    main(args)