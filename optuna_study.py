"""
Optuna Study per l'Ottimizzazione degli Iperparametri del Portafoglio DRL (Dati Reali)

Usage:
    py optuna_study.py --n_trials 50 --total_episodes 50 --fast_mode  (MODALITA COMPLETA)
    py optuna_study.py --n_trials 50 --total_episodes 50 (MODALITA FAST)

Descrizione:
    Questo script utilizza Optuna per ottimizzare gli iperparametri del modello DRL usando dati reali.
    Vengono caricati i dati dai file CSV e JSON dal path specificato. I dati vengono allineati per
    avere lo stesso intervallo di date. La funzione obiettivo esegue un training "light" (con un numero ridotto
    di timestep per episodio e pretraining steps) e valuta la performance, ad esempio la media delle ricompense finali.
    
    I parametri ottimizzati includono:
      - lr_actor, lr_critic (learning rates)
      - tau_actor, tau_critic (parametri di aggiornamento soft)
      - batch_size
      - Dimensioni dei layer del critic
      - Parametri del processo OU (theta, sigma)
    
    Il flag --fast_mode, se abilitato, riduce ulteriormente il numero di timestep per episodio e i pretraining steps.
    
    I risultati dello studio vengono salvati in "optuna_study_results.csv".
    
Note:
    - Assicurati che i moduli 'portfolio_agent' e 'portfolio_env' siano nel PYTHONPATH.
    - La struttura dei dati reali deve essere la seguente:
        BASE_PATH\
            json\
                ARKG_norm_params.json
                IBB_norm_params.json
                ...
            ARKG\
                ARKG_normalized.csv
            IBB\
                IBB_normalized.csv
            ...
"""

import optuna
import numpy as np
import torch
import logging
import argparse
import os
import pandas as pd

from portfolio_agent import PortfolioAgent
from portfolio_env import PortfolioEnvironment

# Configuriamo il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path per i dati reali
BASE_PATH = r'C:\Users\Administrator\Desktop\DRL PORTFOLIO\NAS Results\Multi_Ticker\Normalized_RL_INPUT'
NORM_PARAMS_PATH_BASE = os.path.join(BASE_PATH, "json")
CSV_PATH_BASE = BASE_PATH

# Lista delle feature (questo determina features_per_asset)
norm_columns = [
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
]
features_per_asset = len(norm_columns)

def load_data_for_tickers(tickers, train_fraction=0.8):
    dfs_train = {}
    dfs_test = {}
    norm_params_paths = {}
    valid_tickers = []
    
    for ticker in tickers:
        norm_params_path = os.path.join(NORM_PARAMS_PATH_BASE, f"{ticker}_norm_params.json")
        csv_path = os.path.join(CSV_PATH_BASE, ticker, f"{ticker}_normalized.csv")
        if not os.path.exists(norm_params_path) or not os.path.exists(csv_path):
            logging.warning(f"File per {ticker} non trovato. Salto il ticker.")
            continue
        
        logging.info(f"Caricamento dati per {ticker}...")
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        train_size = int(len(df) * train_fraction)
        dfs_train[ticker] = df.iloc[:train_size].copy()
        dfs_test[ticker] = df.iloc[train_size:].copy()
        norm_params_paths[ticker] = norm_params_path
        valid_tickers.append(ticker)
        logging.info(f"Dataset per {ticker} caricato: {len(df)} righe")
    return dfs_train, dfs_test, norm_params_paths, valid_tickers

def align_dataframes(dfs):
    aligned_dfs = {}
    if all('date' in df.columns for df in dfs.values()):
        start_date = max(df['date'].min() for df in dfs.values())
        end_date = min(df['date'].max() for df in dfs.values())
        logging.info(f"Intervallo di date comune: {start_date} - {end_date}")
        for ticker, df in dfs.items():
            aligned_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
            aligned_df = aligned_df.sort_values('date')
            aligned_dfs[ticker] = aligned_df
        lengths = [len(df) for df in aligned_dfs.values()]
        if len(set(lengths)) > 1:
            min_length = min(lengths)
            logging.info(f"Troncamento a {min_length} righe per uniformare i DataFrame.")
            for ticker in aligned_dfs:
                aligned_dfs[ticker] = aligned_dfs[ticker].iloc[:min_length].copy()
    else:
        min_rows = min(len(df) for df in dfs.values())
        for ticker, df in dfs.items():
            aligned_dfs[ticker] = df.iloc[:min_rows].copy()
    return aligned_dfs

# Parametri di tuning di default
DEFAULT_TOTAL_EPISODES = 50
DEFAULT_FAST_MODE = True
PRETRAIN = 256

def objective(trial):
    try:
        # Suggerimenti per gli iperparametri
        lr_actor = trial.suggest_float('lr_actor', 1e-6, 1e-3, log=True)
        lr_critic = trial.suggest_float('lr_critic', 1e-6, 1e-3, log=True)
        tau_actor = trial.suggest_float('tau_actor', 1e-3, 1e-1, log=True)
        tau_critic = trial.suggest_float('tau_critic', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        fc1_units_critic = trial.suggest_categorical('fc1_units_critic', [256, 512, 1024])
        fc2_units_critic = trial.suggest_categorical('fc2_units_critic', [128, 256, 512])
        fc3_units_critic = trial.suggest_categorical('fc3_units_critic', [64, 128, 256])
        theta = trial.suggest_float('theta', 0.05, 0.2)
        sigma = trial.suggest_float('sigma', 0.1, 0.5)

        # Carica i dati reali per i ticker
        tickers = ["ARKG", "IBB", "IHI", "IYH", "XBI", "VHT"]
        dfs_train, dfs_test, norm_params_paths, valid_tickers = load_data_for_tickers(tickers)
        aligned_dfs_train = align_dataframes(dfs_train)

        if args.fast_mode:
            logging.info("FAST MODE: training light")
            T_training = 50
            pretrain_steps = 128
        else:
            logging.info("FULL MODE: training completo")
            T_training = min(1000, min(len(df) for df in aligned_dfs_train.values()) - 1)
            pretrain_steps = PRETRAIN

        env = PortfolioEnvironment(
            tickers=tickers,
            sigma=0.1,
            theta=0.1,
            T=T_training,
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
            dfs=aligned_dfs_train,
            max_step=T_training,
            norm_params_paths=norm_params_paths,
            norm_columns=norm_columns,
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

        agent = PortfolioAgent(
            num_assets=len(tickers),
            memory_type="prioritized",
            batch_size=batch_size,
            max_step=env.max_step,
            theta=theta,
            sigma=sigma,
            use_enhanced_actor=True,
            use_batch_norm=True
        )

        total_episodes = args.total_episodes

        results = agent.train(
            env=env,
            total_episodes=total_episodes,
            tau_actor=tau_actor,
            tau_critic=tau_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            weight_decay_actor=1e-6,
            weight_decay_critic=1e-5,
            total_steps=pretrain_steps,
            weights="temp_weights/",
            freq=5,
            decay_rate=0.0,
            explore_stop=0.1,
            tensordir="runs/temp/",
            learn_freq=5,
            progress="none",
            features_per_asset=features_per_asset,  # Ora features_per_asset è impostato correttamente
            encoding_size=32,
            clip_grad_norm=1.0,
            checkpoint_path="temp_weights/",
            checkpoint_freq=10,
            resume_from=None,
            early_stop_patience=5
        )
        
        mean_reward = np.mean(list(results['final_rewards']))
        logging.info(f"Trial {trial.number} -> Mean Reward: {mean_reward:.4f}")
        return mean_reward

    except Exception as e:
        logging.error(f"Errore nel trial {trial.number}: {e}")
        return -np.inf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Study per l'ottimizzazione degli iperparametri del Portafoglio DRL (Dati Reali)")
    parser.add_argument("--n_trials", type=int, default=50, help="Numero di trial da eseguire")
    parser.add_argument("--total_episodes", type=int, default=DEFAULT_TOTAL_EPISODES, help="Numero di episodi per ogni trial")
    parser.add_argument("--fast_mode", action="store_false", help="Disabilita fast mode (utilizza impostazioni complete).")
    args = parser.parse_args()

    if args.fast_mode:
        logging.info("Fast mode abilitata: impostazioni light per il training.")
    else:
        logging.info("Fast mode disabilitata: utilizzo delle impostazioni standard per il training.")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    
    logging.info("Miglior trial:")
    best_trial = study.best_trial
    logging.info(f"  Value: {best_trial.value}")
    logging.info("  Params:")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")
    
    df_trials = study.trials_dataframe()
    output_csv = "optuna_study_results.csv"
    df_trials.to_csv(output_csv, index=False)
    logging.info(f"I risultati dello studio sono stati salvati in '{output_csv}'.")