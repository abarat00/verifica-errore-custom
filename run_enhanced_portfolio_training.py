"""
Enhanced Portfolio Training with Advanced DRL Features

Usage:
    python run_enhanced_portfolio_training.py [--options]
    python run_enhanced_portfolio_training.py --profile active --episodes 100 --output_dir results/active_trader
    python run_enhanced_portfolio_training.py --profile longterm --use_calendar --calendar_path calendar.csv

Description:
    This script runs the enhanced portfolio training with the following advanced features:
    
    1. Risk Management: Uses Conditional Value at Risk (CVaR) and improved diversification metrics
    2. Adaptive Exploration: Dynamically adjusts exploration noise based on performance
    3. Market Regime Detection: Identifies and adapts to different market conditions
    4. Transaction Cost Optimization: Calculates optimal trade sizes
    5. Distributional RL: Models the full distribution of returns instead of just the means
    6. Meta-Learning: Quickly adapts to new market data with MAML-inspired techniques
    7. Trading Profiles: Predefined configurations for different trading strategies
    
    Options allow for selecting which enhancements to use and their specific configurations.
    
Basic Arguments:
    --output_dir TEXT           Directory for saving models and results (default: results/enhanced_portfolio)
    --resume TEXT               Path to checkpoint for resuming training
    --episodes INT              Number of training episodes (default: 200)
    --test_ratio FLOAT          Ratio of data to use for testing (default: 0.2)
    --commission_rate FLOAT     Trading commission rate (default: 0.0025)
    --free_trades INT           Number of free trades per month (default: 10)
    --learning_setup TEXT       Either 'full' or 'fast' for different training setups

Enhancement Features:
    --use_adaptive_exploration  Use adaptive exploration strategy
    --use_market_regimes        Use market regime detection
    --use_distributional        Use distributional critic
    --use_meta_learning         Use meta-learning for faster adaptation
    --use_calendar              Use financial calendar data (requires calendar CSV)
    --calendar_path TEXT        Path to financial events calendar CSV
    --use_backtest_framework    Use backtesting framework for evaluation

Trading Profiles:
    --profile TEXT              Trading profile to use: 'active', 'longterm', 'momentum', or 'custom'
    --override_reward           Override reward function parameters with custom values
    --reward_return_weight      Weight for absolute return in reward calculation
    --position_stability_factor Factor for position stability bonus
    --diversification_factor    Factor for diversification bonus
    --trading_activity_factor   Factor for incentivizing active trading
    --trend_following_factor    Factor for incentivizing trend following

Predefined Profiles:
    active:   High trading frequency, realistic commissions, emphasis on returns
    longterm: Low turnover, emphasis on diversification and stability
    momentum: Medium trading frequency, focus on trend following, timing focus
    custom:   User-defined parameters (default if no profile specified)
"""
#py run_enhanced_portfolio_training.py 
# --use_adaptive_exploration 
# --use_market_regimes --learning_setup fast

#BASELINE FULL RUN WITH COMMISSION
#py run_enhanced_portfolio_training.py --learning_setup full --output_dir results/baseline_full --episodes 3 --use_calendar --calendar_path calendar.csv

#BASELINE FULL RUN WITHOUT COMMISSION
#py run_enhanced_portfolio_training.py --learning_setup full --output_dir results/baseline_full --episodes 3 --free_trades 10000000 --commission_rate 0.0 --use_calendar --calendar_path calendar.csv

#INTEGRAZIONE CALENDAR
#py run_enhanced_portfolio_training.py --learning_setup full --output_dir results/calendar_training --episodes 3 --use_calendar --calendar_path calendar.csv

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import deque
import argparse

# Import our modules
from portfolio_agent import PortfolioAgent, AdaptiveExploration
from portfolio_env import PortfolioEnvironment
from portfolio_models import PortfolioCritic, EnhancedPortfolioActor, DistributionalCritic, MAMLPortfolioActor
from market_regime import MarketRegimeDetector
from financial_calendar import FinancialCalendar
from portfolio_construction import HybridPortfolioConstructor
from backtesting import BacktestFramework
from financial_calendar import FinancialCalendar

# List of tickers to use in the portfolio
TICKERS = ["XLF", "XLE", "XLK", "IHI", "XLY"]

# Base configuration
BASE_PATH = 'C:\\Users\\Administrator\\Desktop\\DRL PORTFOLIO\\NAS Results\\Multi_Ticker\\Normalized_RL_INPUT\\'
NORM_PARAMS_PATH_BASE = f'{BASE_PATH}json\\'
CSV_PATH_BASE = f'{BASE_PATH}'

# Features to use
norm_columns = [
    "open", "volume", "change", "day", "week", "adjCloseGold", "adjCloseSpy",
    "Credit_Spread", #"Log_Close",
    "m_plus", "m_minus", "drawdown", "drawup",
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

def check_file_exists(file_path):
    """Verify if a file exists and print an appropriate message."""
    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {file_path}")
        return False
    return True

def load_data_for_tickers(tickers, train_fraction=0.8):
    """
    Load and prepare data for all tickers.
    
    Parameters:
    - tickers: list of tickers to load
    - train_fraction: fraction of data to use for training (0.8 = 80%)
    
    Returns:
    - dfs_train: dict of DataFrames for training
    - dfs_test: dict of DataFrames for testing
    - norm_params_paths: dict of paths to normalization parameters
    - valid_tickers: list of tickers that were successfully loaded
    """
    dfs_train = {}
    dfs_test = {}
    norm_params_paths = {}
    valid_tickers = []
    
    for ticker in tickers:
        norm_params_path = f'{NORM_PARAMS_PATH_BASE}{ticker}_norm_params.json'
        csv_path = f'{CSV_PATH_BASE}{ticker}\\{ticker}_normalized.csv'
        
        # Verify file existence
        if not (check_file_exists(norm_params_path) and check_file_exists(csv_path)):
            print(f"Skipping ticker {ticker} due to missing files")
            continue
        
        # Load dataset
        print(f"Loading data for {ticker}...")
        df = pd.read_csv(csv_path)
        
        # Check for all required columns
        missing_cols = [col for col in norm_columns if col not in df.columns]
        if missing_cols:
            print(f"Skipping ticker {ticker}. Missing columns: {missing_cols}")
            continue
        
        # Sort dataset by date (if present)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        # Split into training and test
        train_size = int(len(df) * train_fraction)
        dfs_train[ticker] = df.iloc[:train_size]
        dfs_test[ticker] = df.iloc[train_size:]
        norm_params_paths[ticker] = norm_params_path
        
        valid_tickers.append(ticker)
        print(f"Dataset for {ticker} loaded: {len(df)} rows")
    
    return dfs_train, dfs_test, norm_params_paths, valid_tickers

def save_results(results, output_dir, tickers, enhancement_config):
    """
    Save training results with information about which enhancements were used.
    """
    try:
        # Extract metrics safely
        if isinstance(results['final_rewards'], deque):
            final_reward = np.mean(list(results['final_rewards'])[-3:]) if len(results['final_rewards']) >= 3 else np.mean(list(results['final_rewards']))
        else:
            final_reward = results['final_rewards']
            
        if isinstance(results['final_portfolio_values'], deque):
            final_portfolio_value = np.mean(list(results['final_portfolio_values'])[-3:]) if len(results['final_portfolio_values']) >= 3 else np.mean(list(results['final_portfolio_values']))
        else:
            final_portfolio_value = results['final_portfolio_values']
            
        if isinstance(results['final_sharpe_ratios'], deque):
            final_sharpe_ratio = np.mean(list(results['final_sharpe_ratios'])[-3:]) if len(results['final_sharpe_ratios']) >= 3 else np.mean(list(results['final_sharpe_ratios']))
        else:
            final_sharpe_ratio = results['final_sharpe_ratios']

        if 'cvar_values' in results and len(results['cvar_values']) > 0:
            final_cvar = np.mean(list(results['cvar_values'])[-3:]) if len(results['cvar_values']) >= 3 else np.mean(list(results['cvar_values']))
        else:
            final_cvar = None
            
        # Create DataFrame
        results_dict = {
            'ticker': [', '.join(tickers)],
            'final_reward': [final_reward],
            'final_portfolio_value': [final_portfolio_value],
            'final_sharpe_ratio': [final_sharpe_ratio],
            'training_timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        
        # Add info about which enhancements were used
        for key, value in enhancement_config.items():
            results_dict[key] = [value]
            
        # Add CVaR if available
        if final_cvar is not None:
            results_dict['final_cvar'] = [final_cvar]
            
        results_df = pd.DataFrame(results_dict)
        
        # Save the DataFrame
        results_file = f'{output_dir}\\training_results.csv'
        if os.path.exists(results_file):
            existing_results = pd.read_csv(results_file)
            updated_results = pd.concat([existing_results, results_df], ignore_index=True)
            updated_results.to_csv(results_file, index=False)
        else:
            results_df.to_csv(results_file, index=False)
        
        print(f"Results saved in: {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
        # Save raw data in pickle format for later analysis
        import pickle
        with open(f'{output_dir}\\raw_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Raw results saved in: {output_dir}\\raw_results.pkl")

def plot_enhanced_performance(results, output_dir, tickers, enhancement_config):
    """Create visualizations of training performance with enhanced metrics."""
    plt.figure(figsize=(15, 12))
    
    # Plot cumulative rewards
    plt.subplot(3, 2, 1)
    plt.plot(results['cum_rewards'])
    plt.title('Cumulative Reward')
    plt.xlabel('Episodes (x5)')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot portfolio values
    plt.subplot(3, 2, 2)
    plt.plot(results['final_portfolio_values'])
    plt.title('Final Portfolio Value')
    plt.xlabel('Episodes')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.3)
    
    # Plot Sharpe ratios
    plt.subplot(3, 2, 3)
    plt.plot(results['final_sharpe_ratios'])
    plt.title('Sharpe Ratio')
    plt.xlabel('Episodes')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot CVaR if available
    if 'cvar_values' in results and len(results['cvar_values']) > 0:
        plt.subplot(3, 2, 4)
        plt.plot(results['cvar_values'])
        plt.title('Conditional Value at Risk (CVaR)')
        plt.xlabel('Episodes')
        plt.ylabel('CVaR')
        plt.grid(True, alpha=0.3)
    else:
        # Plot diversification metrics if no CVaR
        if 'diversification_metrics' in results and len(results['diversification_metrics']) > 0:
            plt.subplot(3, 2, 4)
            plt.plot(results['diversification_metrics'])
            plt.title('Portfolio Diversification')
            plt.xlabel('Episodes')
            plt.ylabel('Diversification Score')
            plt.grid(True, alpha=0.3)
    
    # Plot regime changes if market regime detection was used
    if enhancement_config.get('use_market_regimes', False) and 'regime_changes' in results:
        plt.subplot(3, 2, 5)
        plt.plot(results['regime_changes'])
        plt.title('Market Regime Changes')
        plt.xlabel('Episodes')
        plt.ylabel('Regime ID')
        plt.grid(True, alpha=0.3)
    else:
        # Plot exploration metrics
        if 'exploration_rates' in results:
            plt.subplot(3, 2, 5)
            plt.plot(results['exploration_rates'])
            plt.title('Exploration Rate')
            plt.xlabel('Episodes')
            plt.ylabel('Exploration Rate')
            plt.grid(True, alpha=0.3)
    
    # Text summary of configurations
    plt.subplot(3, 2, 6)
    plt.axis('off')
    text = "Enhanced Portfolio Configuration:\n\n"
    for key, value in enhancement_config.items():
        text += f"- {key}: {value}\n"
    text += f"\nTickers: {', '.join(tickers)}\n"
    text += f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    plt.text(0.1, 0.5, text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}\\enhanced_performance.png')
    print(f"Performance visualization saved in: {output_dir}\\enhanced_performance.png")

def align_dataframes(dfs):
    """
    Align DataFrames to have the same date range and number of rows.
    """
    aligned_dfs = {}
    
    # Find common date range
    if all('date' in df.columns for df in dfs.values()):
        # Find most recent start date
        start_date = max(df['date'].min() for df in dfs.values())
        # Find earliest end date
        end_date = min(df['date'].max() for df in dfs.values())
        
        print(f"Common date range: {start_date} - {end_date}")
        
        # Filter and align each DataFrame
        for ticker, df in dfs.items():
            aligned_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
            # Make sure dates are sorted
            aligned_df = aligned_df.sort_values('date')
            aligned_dfs[ticker] = aligned_df
        
        # Check that all aligned DataFrames have the same number of rows
        lengths = [len(df) for df in aligned_dfs.values()]
        if len(set(lengths)) > 1:
            print(f"WARNING: Aligned DataFrames have different lengths: {lengths}")
            # Find minimum length
            min_length = min(lengths)
            print(f"Truncating to {min_length} rows...")
            # Truncate all DataFrames to the same length
            for ticker in aligned_dfs:
                aligned_dfs[ticker] = aligned_dfs[ticker].iloc[:min_length].copy()
    else:
        # If no 'date' columns, use the minimum number of rows
        min_rows = min(len(df) for df in dfs.values())
        for ticker, df in dfs.items():
            aligned_dfs[ticker] = df.iloc[:min_rows].copy()
    
    # Final length check
    lengths = [len(df) for df in aligned_dfs.values()]
    print(f"Aligned DataFrame lengths: {lengths}")
    
    return aligned_dfs

def create_enhanced_agent(args, num_assets, max_steps, features_per_asset, output_dir, config=None, **kwargs):
    """
    Create an agent with the selected enhancements based on argument flags.
    """
    # Usa i parametri dal dizionario di configurazione se disponibile
    if config and 'init' in config:
        agent_config = config['init'].copy()
    else:
        # Fallback su kwargs
        agent_config = {
            'memory_type': "prioritized",
            'batch_size': 256,
            'theta': 0.03,
            'sigma': kwargs.get('exploration_sigma', 0.2),
            'use_enhanced_actor': True,
            'use_batch_norm': True
        }
    
    # Aggiungi parametri non configurabili
    agent_config.update({
        'num_assets': num_assets,
        'max_step': max_steps
    })
    
    # Crea l'agente
    agent = PortfolioAgent(**agent_config)
    
    # Setup adaptive exploration if requested
    if args.use_adaptive_exploration:
        print("Initializing adaptive exploration strategy...")
        agent.noise = AdaptiveExploration(
            action_size=num_assets, 
            theta=0.1, 
            sigma=0.2, 
            min_sigma=0.05
        )
    
    # Salva la configurazione
    agent_config_path = f'{output_dir}\\agent_config.json'
    with open(agent_config_path, 'w') as f:
        import json
        json.dump(agent_config, f, indent=4)
    
    return agent

# Modifica la funzione create_enhanced_environment per risolvere il problema di serializzazione
def create_enhanced_environment(args, valid_tickers, aligned_dfs_train, norm_params_paths, max_steps, output_dir, config=None, calendar=None, **kwargs):
    """
    Create the portfolio environment with enhanced features based on argument flags.
    """
    if calendar is None and args.use_calendar and args.calendar_path and os.path.exists(args.calendar_path):
        print(f"Loading financial calendar from {args.calendar_path}...")
        calendar = FinancialCalendar(args.calendar_path)
        print(f"Loaded events from calendar.")
    
    # Base configuration
    env_config = {
        'tickers': valid_tickers,
        'sigma': 0.1,
        'theta': 0.1,
        'T': max_steps,
        'lambd': 0.05,
        'psi': 0.2,
        'cost': "trade_l1",
        'dfs': aligned_dfs_train,
        'max_step': max_steps,
        'norm_params_paths': norm_params_paths,
        'norm_columns': norm_columns,
        'initial_capital': 100000,
        'risk_free_rate': 0.02,
        'use_sortino': True,
        'target_return': 0.05,
        'calendar': calendar,
        
        # Valori default che posso essere sovrascritti
        'max_pos_per_asset': 2.0,
        'max_portfolio_pos': 6.0,
        'squared_risk': False,
        'penalty': "tanh",
        'alpha': 3,
        'beta': 3,
        'clip': True,
        'scale_reward': 5,
        'free_trades_per_month': args.free_trades,
        'commission_rate': args.commission_rate,
        'min_commission': 1.0 if args.commission_rate > 0 else 0.0,
        'trading_frequency_penalty_factor': 0.1,
        'position_stability_bonus_factor': 0.2,
        'correlation_penalty_factor': 0.15,
        'diversification_bonus_factor': 0.25,
        'inactivity_penalty': 0.0,
        'reward_return_weight': 1.0,
        'trend_following_factor': 0.0
    }
    
    # Aggiornamento con parametri dal profilo
    if config and 'env' in config:
        env_params = config['env']
        for key, value in env_params.items():
            if key in env_config:
                env_config[key] = value
        
        # Calcolo speciale per min_commission in base a commission_rate
        if 'commission_rate' in env_params:
            env_config['min_commission'] = 1.0 if env_params['commission_rate'] > 0 else 0.0
    
    # Create the environment with the configuration
    env = PortfolioEnvironment(**env_config)
    
    # If using market regime detection, prepare the detector
    if args.use_market_regimes:
        print("Initializing market regime detector...")
        # This will be used in the main training loop
    
    # Save the environment configuration (omitting non-serializable parts)
    env_config_path = f'{output_dir}\\env_config.json'
    with open(env_config_path, 'w') as f:
        import json
        # Create a copy with only serializable items
        serializable_config = {}
        for k, v in env_config.items():
            if k == 'dfs':
                # Skip DataFrames
                continue
            elif k == 'norm_params_paths':
                # Convert dict to a simple list of paths
                serializable_config[k] = list(v.values()) if v else []
            elif k == 'calendar':
                # Skip calendar object
                continue
            elif isinstance(v, (int, float, bool, str, list, type(None))):
                serializable_config[k] = v
            else:
                # Convert other non-serializable objects to strings
                serializable_config[k] = str(v)
                
        json.dump(serializable_config, f, indent=4)
    
    return env

def monitor_enhanced_metrics(env, agent, episode, i, writer, regime_detector=None, calendar=None, results=None):
    """
    Track enhanced metrics during training, especially those related to the new features.
    """
    # If first call, initialize the results dictionary
    if results is None:
        results = {
            'cvar_values': [],
            'regime_changes': [],
            'exploration_rates': [],
            'diversification_metrics': [],
            'meta_learning_adaptations': []
        }
    
    # Calculate and record CVaR
    cvar = env.calculate_conditional_value_at_risk(confidence_level=0.95)
    results['cvar_values'].append(cvar)
    if writer:
        writer.add_scalar("Risk/CVaR", cvar, i)
    
    # If using adaptive exploration, record the current sigma
    if hasattr(agent.noise, 'adapt_sigma'):
        # Assuming we have some performance metric to adapt to
        portfolio_metrics = env.get_real_portfolio_metrics()
        current_sharpe = portfolio_metrics['sharpe_ratio']
        # Adapt sigma based on Sharpe ratio (target is something like 1.0 for good performance)
        current_sigma = agent.noise.adapt_sigma(max(0.01, current_sharpe), 1.0)
        results['exploration_rates'].append(current_sigma)
        if writer:
            writer.add_scalar("Exploration/Current_Sigma", current_sigma, i)
    
    # If using market regime detection
    if regime_detector is not None:
        # Get price history to detect regime
        ticker = env.tickers[0]  # Example: use first ticker for simplicity
        price_history = np.array(env.price_history[ticker])
        if len(price_history) > regime_detector.window_size:
            current_regime = regime_detector.detect_regime(price_history)
            results['regime_changes'].append(current_regime)
            if writer:
                writer.add_scalar("Market/Current_Regime", current_regime, i)
    
    # Record diversification metrics
    diversification = env.calculate_diversification_bonus() / env.diversification_bonus_factor
    results['diversification_metrics'].append(diversification)
    if writer:
        writer.add_scalar("Portfolio/Diversification", diversification, i)
    
    # If using financial calendar
    if calendar is not None:
        # Get current date
        if all('date' in df.columns for df in env.dfs.values()):
            current_date = list(env.dfs.values())[0]['date'].iloc[env.current_index]
            upcoming_events = calendar.get_upcoming_events(current_date)
            if upcoming_events and writer:
                # Count events by importance
                importance_count = {}
                for event in upcoming_events:
                    imp = event['importance']
                    importance_count[imp] = importance_count.get(imp, 0) + 1
                for imp, count in importance_count.items():
                    writer.add_scalar(f"Calendar/Events_{imp}", count, i)
    
    return results

# In run_enhanced_portfolio_training.py

def add_profile_arguments(parser):
    """Aggiunge argomenti relativi ai profili di trading."""
    profile_group = parser.add_argument_group('Profili di trading')
    
    # Argomento per selezionare un profilo predefinito
    profile_group.add_argument('--profile', type=str, choices=['active', 'longterm', 'momentum', 'custom'], 
                             default='custom', help='Profilo di trading predefinito da utilizzare')
    
    # Per consentire override di parametri specifici
    profile_group.add_argument('--override_reward', action='store_true',
                             help='Override della funzione di ricompensa con parametri personalizzati')
    profile_group.add_argument('--reward_return_weight', type=float, default=None,
                             help='Peso del rendimento assoluto nella ricompensa')
    profile_group.add_argument('--position_stability_factor', type=float, default=None,
                             help='Fattore di bonus per stabilità delle posizioni')
    profile_group.add_argument('--diversification_factor', type=float, default=None,
                             help='Fattore di bonus per diversificazione')
    profile_group.add_argument('--trading_activity_factor', type=float, default=None,
                             help='Fattore di incentivo per trading attivo')
    profile_group.add_argument('--trend_following_factor', type=float, default=None,
                             help='Fattore di incentivo per seguire trend di mercato')
    profile_group.add_argument('--inactivity_penalty', type=float, default=None,
                           help='Penalità per inattività di trading')
    profile_group.add_argument('--max_pos_per_asset', type=float, default=None,
                         help='Posizione massima consentita per singolo asset')

def get_profile_config(args):
    """
    Crea una configurazione basata sul profilo selezionato con struttura nidificata.
    """
    profiles = {
        'active': {
            # Parametri per __init__
            'init': {
                'memory_type': "prioritized",
                'batch_size': 256,
                'theta': 0.03,
                'sigma': 0.3,  # Esplorazione più alta
                'use_enhanced_actor': True,
                'use_batch_norm': True
            },
            # Parametri per train()
            'train': {
                'lr_actor': 1e-4,  # Learning rate più alto
                'lr_critic': 1e-3,
                'weight_decay_actor': 1e-6,
                'weight_decay_critic': 1e-5,
                'decay_rate': 1e-6,
                'explore_stop': 0.1,
                'encoding_size': 32,
                'clip_grad_norm': 1.2,
                'early_stop_patience': 10  # Minore pazienza
            },
            # Parametri per environment
            'env': {
                'commission_rate': 0.0025,  # Commission_rate realistica
                'free_trades': 5,  # Poche operazioni gratuite
                'max_pos_per_asset': 1.0,  # Posizione massima ridotta
                'max_portfolio_pos': 3.0,  # Esposizione massima ridotta
                'position_stability_factor': 0.01,  # Basso bonus per stabilità
                'diversification_factor': 0.05,  # Ridotto bonus diversificazione
                'trading_activity_factor': 0.1,  # Alto incentivo per trading attivo
                'trend_following_factor': 0.05,  # Moderato incentivo per trend following
                'reward_return_weight': 5.0,  # Alto peso per rendimento
                'inactivity_penalty': 0.001,  # Penalità per inattività
                'target_sharpe': 1.2  # Target Sharpe più aggressivo
            }
        },
        'longterm': {
            # Parametri per __init__
            'init': {
                'memory_type': "prioritized",
                'batch_size': 256,
                'theta': 0.03,
                'sigma': 0.1,  # Esplorazione più bassa
                'use_enhanced_actor': True,
                'use_batch_norm': True
            },
            # Parametri per train()
            'train': {
                'lr_actor': 1e-4,  # Learning rate più alto
                'lr_critic': 1e-3,
                'weight_decay_actor': 1e-6,
                'weight_decay_critic': 1e-5,
                'decay_rate': 5e-7,
                'explore_stop': 0.1,
                'encoding_size': 32,
                'clip_grad_norm': 1.0,
                'early_stop_patience': 20  # Maggiore pazienza
            },
            # Parametri per environment
            'env': {
                'commission_rate': 0.001,  # Commission_rate bassa
                'free_trades': 3,  # Pochissime operazioni gratuite
                'max_pos_per_asset': 2.0,  # Posizione massima standard
                'max_portfolio_pos': 5.0,  # Esposizione massima standard
                'position_stability_factor': 0.2,  # Alto bonus per stabilità
                'diversification_factor': 0.2,  # Alto bonus diversificazione
                'trading_activity_factor': 0.0,  # Nessun incentivo per trading attivo
                'trend_following_factor': 0.0,  # Nessun incentivo per trend following
                'reward_return_weight': 2.0,  # Moderato peso per rendimento
                'inactivity_penalty': 0.0,  # Nessuna penalità per inattività
                'target_sharpe': 1.0  # Target Sharpe moderato
            }
        },
        'momentum': {
            # Parametri per __init__
            'init': {
                'memory_type': "prioritized",
                'batch_size': 256,
                'theta': 0.03,
                'sigma': 0.2,  # Esplorazione media
                'use_enhanced_actor': True,
                'use_batch_norm': True
            },
            # Parametri per train()
            'train': {
                'lr_actor': 1e-4,  # Learning rate più alto
                'lr_critic': 1e-3,
                'weight_decay_actor': 1e-6,
                'weight_decay_critic': 1e-5,
                'decay_rate': 5e-7,
                'explore_stop': 0.1,
                'encoding_size': 32,
                'clip_grad_norm': 1.0,
                'early_stop_patience': 15  # Pazienza media
            },
            # Parametri per environment
            'env': {
                'commission_rate': 0.002,  # Commission_rate media
                'free_trades': 8,  # Numero medio di operazioni gratuite
                'max_pos_per_asset': 1.5,  # Posizione massima media
                'max_portfolio_pos': 4.0,  # Esposizione massima media
                'position_stability_factor': 0.05,  # Basso bonus per stabilità
                'diversification_factor': 0.1,  # Moderato bonus diversificazione
                'trading_activity_factor': 0.05,  # Moderato incentivo per trading attivo
                'trend_following_factor': 0.2,  # Alto incentivo per trend following
                'reward_return_weight': 3.0,  # Moderato peso per rendimento
                'inactivity_penalty': 0.0005,  # Piccola penalità per inattività
                'target_sharpe': 0.8  # Target Sharpe più basso
            }
        },
        'custom': {
            # Parametri per __init__
            'init': {
                'memory_type': "prioritized",
                'batch_size': 256,
                'theta': 0.03,
                'sigma': 0.2,
                'use_enhanced_actor': True,
                'use_batch_norm': True
            },
            # Parametri per train()
            'train': {
                'lr_actor': 1e-4,
                'lr_critic': 1e-3,
                'weight_decay_actor': 1e-6,
                'weight_decay_critic': 1e-5,
                'decay_rate': 5e-7,
                'explore_stop': 0.1,
                'encoding_size': 32,
                'clip_grad_norm': 1.0,
                'early_stop_patience': 15
            },
            # Parametri per environment
            'env': {
                'commission_rate': args.commission_rate,
                'free_trades': args.free_trades,
                'max_pos_per_asset': 2.0,
                'max_portfolio_pos': 6.0,
                'position_stability_factor': 0.2,
                'diversification_factor': 0.25,
                'trading_activity_factor': 0.0,
                'trend_following_factor': 0.0,
                'reward_return_weight': 1.0,
                'inactivity_penalty': 0.0,
                'target_sharpe': 0.5
            }
        }
    }
    
    # Ottieni la configurazione di base dal profilo selezionato
    config = {
        'init': profiles[args.profile]['init'].copy(),
        'train': profiles[args.profile]['train'].copy(),
        'env': profiles[args.profile]['env'].copy()
    }
    
    # Applica gli override specificati come argomenti
    if args.commission_rate is not None:
        config['env']['commission_rate'] = args.commission_rate
    if args.free_trades is not None:
        config['env']['free_trades'] = args.free_trades
    
    # Override di parametri specifici della funzione di ricompensa
    if args.override_reward:
        if args.reward_return_weight is not None:
            config['env']['reward_return_weight'] = args.reward_return_weight
        if args.position_stability_factor is not None:
            config['env']['position_stability_factor'] = args.position_stability_factor
        if args.diversification_factor is not None:
            config['env']['diversification_factor'] = args.diversification_factor
        if args.trading_activity_factor is not None:
            config['env']['trading_activity_factor'] = args.trading_activity_factor
        if args.trend_following_factor is not None:
            config['env']['trend_following_factor'] = args.trend_following_factor
        if args.inactivity_penalty is not None:
            config['env']['inactivity_penalty'] = args.inactivity_penalty
        if args.max_pos_per_asset is not None:
            config['env']['max_pos_per_asset'] = args.max_pos_per_asset
        
    return config


def main(args):
    """Main function for enhanced portfolio training."""
    # Create output directories
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}\\weights', exist_ok=True)
    os.makedirs(f'{output_dir}\\test', exist_ok=True)
    os.makedirs(f'{output_dir}\\analysis', exist_ok=True)
    
    # Ottieni la configurazione in base al profilo selezionato
    profile_config = get_profile_config(args)

    # Salva la configurazione del profilo
    with open(f'{output_dir}\\profile_config.json', 'w') as f:
        import json
        json.dump(profile_config, f, indent=4)

    # 1. Load and prepare data
    print("Loading data for all tickers...")
    dfs_train, dfs_test, norm_params_paths, valid_tickers = load_data_for_tickers(
        TICKERS, train_fraction=1-args.test_ratio)
    
    if not valid_tickers:
        print("No valid tickers found. Exiting.")
        return
    
    print(f"Valid tickers: {valid_tickers}")
    
    # Align DataFrames for training and testing
    print("Aligning DataFrames...")
    aligned_dfs_train = align_dataframes(dfs_train)
    aligned_dfs_test = align_dataframes(dfs_test)
    
    # Save aligned test DataFrames for future reference
    for ticker, df in aligned_dfs_test.items():
        df.to_csv(f'{output_dir}\\test\\{ticker}_test_aligned.csv', index=False)
    
    # Load calendar once if requested
    calendar = None
    if args.use_calendar and args.calendar_path:
        if os.path.exists(args.calendar_path):
            print(f"Loading financial calendar from {args.calendar_path}...")
            calendar = FinancialCalendar(args.calendar_path)
            print(f"Loaded {sum(len(events) for events in calendar.events.values())} events")
        else:
            print(f"WARNING: Calendar file {args.calendar_path} not found. Running without calendar.")

    # 2. Create the enhanced portfolio environment
    print("Initializing the enhanced portfolio environment...")
    max_steps = min(1000, min(len(df) for df in aligned_dfs_train.values()) - 10)
    
    # Passa l'intero oggetto config alla funzione
    env = create_enhanced_environment(
        args, valid_tickers, aligned_dfs_train, 
        norm_params_paths, max_steps, output_dir,
        config=profile_config,  # Passa l'intero oggetto di configurazione
        calendar=calendar
    )

    env.log_state_dimensions()

    # 2b. Inizializza con posizioni neutre invece che casuali
    env.positions = np.zeros(env.num_assets)

    # 3. Initialize the enhanced agent
    print("Initializing the enhanced portfolio agent...")
    num_assets = len(valid_tickers)
    agent = create_enhanced_agent(
        args, num_assets, max_steps, 
        len(norm_columns), output_dir,
        config=profile_config  # Passa l'intero oggetto di configurazione
    )
    
    # 4. Initialize additional components based on arguments
    regime_detector = None
    
    if args.use_market_regimes:
        regime_detector = MarketRegimeDetector(window_size=60, n_regimes=3)
    
    # 5. Create a dictionary of which enhancements are being used
    enhancement_config = {
        'use_adaptive_exploration': args.use_adaptive_exploration,
        'use_market_regimes': args.use_market_regimes,
        'use_distributional': args.use_distributional,
        'use_meta_learning': args.use_meta_learning,
        'use_calendar': args.use_calendar and args.calendar_path is not None,
        'commission_rate': profile_config['env']['commission_rate'],
        'free_trades': profile_config['env']['free_trades'],
        'learning_setup': args.learning_setup,
        'profile': args.profile,
        'position_stability_factor': profile_config['env']['position_stability_factor'],
        'diversification_factor': profile_config['env']['diversification_factor'],
        'trading_activity_factor': profile_config['env'].get('trading_activity_factor', 0.0),
        'trend_following_factor': profile_config['env'].get('trend_following_factor', 0.0),
        'reward_return_weight': profile_config['env']['reward_return_weight'],
        'inactivity_penalty': profile_config['env'].get('inactivity_penalty', 0.0),
        'max_pos_per_asset': profile_config['env']['max_pos_per_asset']

    }
    
    # Print the current configuration
    print("\n=== Enhanced Portfolio Training Configuration ===")
    for key, value in enhancement_config.items():
        print(f"{key}: {value}")
    print("====================================================\n")
    
    # 6. Determine training parameters based on learning setup
    if args.learning_setup == 'fast':
        print("Using FAST learning setup (quicker but less thorough)")
        training_config = {
            'total_episodes': args.episodes,
            'tau_actor': 0.01,
            'tau_critic': 0.03,
            'weight_decay_actor': 1e-6,
            'weight_decay_critic': 1e-5,
            'total_steps': 1000,  # Pretraining steps
            'fc1_units_actor': 256,
            'fc2_units_actor': 128,
            'fc3_units_actor': 64,
            'fc1_units_critic': 512,
            'fc2_units_critic': 256,
            'fc3_units_critic': 128,
            'learn_freq': 10,
            'early_stop_patience': 20
        }
    else:  # 'full'
        print("Using FULL learning setup (more thorough but slower)")
        training_config = {
            'total_episodes': args.episodes,
            'tau_actor': 0.01,
            'tau_critic': 0.01,
            'weight_decay_actor': 1e-6,
            'weight_decay_critic': 1e-5,
            'total_steps': 3000,  # Pretraining steps
            'fc1_units_actor': 512,
            'fc2_units_actor': 256,
            'fc3_units_actor': 128,
            'fc1_units_critic': 1024,
            'fc2_units_critic': 512,
            'fc3_units_critic': 256,
            'learn_freq': 5,
            'early_stop_patience': 15
        }
    
    # Aggiungi i parametri dal profilo
    training_config.update({
        'lr_actor': profile_config['train']['lr_actor'],
        'lr_critic': profile_config['train']['lr_critic'],
        'decay_rate': profile_config['train']['decay_rate'],
        'explore_stop': profile_config['train']['explore_stop'],
        'encoding_size': profile_config['train']['encoding_size'],
        'clip_grad_norm': profile_config['train']['clip_grad_norm']
    })
    
    # 7. Setup distributional critic if requested
    if args.use_distributional:
        print("Using distributional critic to model full return distribution...")
        # We will replace the critic later in the training loop
    
    # 8. Setup meta-learning actor if requested
    if args.use_meta_learning:
        print("Using meta-learning actor for faster adaptation...")
        # Will be initialized during training
    
    # 9. Start the training
    print(f"Starting enhanced training with {num_assets} assets...")
    
    # Calculate features_per_asset for the EnhancedPortfolioActor
    features_per_asset = len(norm_columns)
    
    # Setup for resuming if specified
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        resume_from = args.resume
    else:
        print("Starting new training run...")
        resume_from = None
    
    # Initialize enhanced metrics tracking
    enhanced_metrics = {
        'cvar_values': [],
        'regime_changes': [],
        'exploration_rates': [],
        'diversification_metrics': [],
        'meta_learning_adaptations': []
    }
    
    # Modify the standard train function to incorporate our enhancements
    class EnhancedTrainingMonitor:
        def __init__(self):
            self.enhanced_metrics = enhanced_metrics
            self.episode_callbacks = []
            
        def register_episode_callback(self, callback):
            self.episode_callbacks.append(callback)
            
        def on_episode_end(self, env, agent, episode, i, writer):
            """Called at the end of each episode to track enhanced metrics"""
            for callback in self.episode_callbacks:
                callback(env, agent, episode, i, writer, self.enhanced_metrics)
    
    # Initialize the training monitor
    monitor = EnhancedTrainingMonitor()
    
    # Register callbacks for different enhancements
    
    # Add CVaR tracking
    def track_cvar(env, agent, episode, i, writer, metrics):
        cvar = env.calculate_conditional_value_at_risk(confidence_level=0.95)
        metrics['cvar_values'].append(cvar)
        writer.add_scalar("Risk/CVaR", cvar, i)
    
    monitor.register_episode_callback(track_cvar)
    
    # Add adaptive exploration tracking if enabled
    if args.use_adaptive_exploration:
        def track_adaptive_exploration(env, agent, episode, i, writer, metrics):
            portfolio_metrics = env.get_real_portfolio_metrics()
            current_sharpe = portfolio_metrics['sharpe_ratio']
            current_sigma = agent.noise.adapt_sigma(max(0.01, current_sharpe), 1.0)
            metrics['exploration_rates'].append(current_sigma)
            writer.add_scalar("Exploration/Current_Sigma", current_sigma, i)
        
        monitor.register_episode_callback(track_adaptive_exploration)
    
    # Add market regime tracking if enabled
    if args.use_market_regimes:
        def track_market_regimes(env, agent, episode, i, writer, metrics):
            ticker = env.tickers[0]  # Example: use first ticker for simplicity
            price_history = np.array(env.price_history[ticker])
            if len(price_history) > regime_detector.window_size:
                current_regime = regime_detector.detect_regime(price_history)
                metrics['regime_changes'].append(current_regime)
                writer.add_scalar("Market/Current_Regime", current_regime, i)
        
        monitor.register_episode_callback(track_market_regimes)
    
    # Add calendar event tracking if enabled
    if args.use_calendar and calendar is not None:
        def track_calendar_events(env, agent, episode, i, writer, metrics):
            if all('date' in df.columns for df in env.dfs.values()):
                current_date = list(env.dfs.values())[0]['date'].iloc[env.current_index]
                upcoming_events = calendar.get_upcoming_events(current_date)
                if upcoming_events:
                    # Count events by importance
                    importance_count = {}
                    for event in upcoming_events:
                        imp = event['importance']
                        importance_count[imp] = importance_count.get(imp, 0) + 1
                    for imp, count in importance_count.items():
                        writer.add_scalar(f"Calendar/Events_{imp}", count, i)
        
        monitor.register_episode_callback(track_calendar_events)
    
    # Add diversification tracking
    def track_diversification(env, agent, episode, i, writer, metrics):
        diversification = env.calculate_diversification_bonus() / env.diversification_bonus_factor
        metrics['diversification_metrics'].append(diversification)
        writer.add_scalar("Portfolio/Diversification", diversification, i)
    
    monitor.register_episode_callback(track_diversification)
    
    # Patch the agent's train method to call our monitor
    original_train = agent.train
    
    def enhanced_train(*args, **kwargs):
        # Store original episode complete logic to hook into it
        #original_episode_complete_branch = None
        # Find and store the episode_complete branch in the train method
        #train_source = original_train.__code__.co_consts
        #for const in train_source:
            #if isinstance(const, str) and "episode_complete" in const:
                #original_episode_complete_branch = const
                #break
        
        # Call the original train method with our additional callback
        results = original_train(*args, **kwargs)
        
        # Add our enhanced metrics to the results
        for key, value in enhanced_metrics.items():
            if value:  # Only add non-empty metrics
                results[key] = value
        
        return results
    
    # Replace the train method with our enhanced version
    agent.train = enhanced_train
    
    # Run the actual training with all enhancements configured
    results = agent.train(
        env=env,
        total_episodes=training_config['total_episodes'],
        tau_actor=training_config['tau_actor'],
        tau_critic=training_config['tau_critic'],
        lr_actor=training_config['lr_actor'],
        lr_critic=training_config['lr_critic'],
        weight_decay_actor=training_config['weight_decay_actor'],
        weight_decay_critic=training_config['weight_decay_critic'],
        total_steps=training_config['total_steps'],
        weights=f'{output_dir}\\weights\\',
        freq=10,
        fc1_units_actor=training_config['fc1_units_actor'],
        fc2_units_actor=training_config['fc2_units_actor'],
        fc3_units_actor=training_config['fc3_units_actor'],
        fc1_units_critic=training_config['fc1_units_critic'],
        fc2_units_critic=training_config['fc2_units_critic'],
        fc3_units_critic=training_config['fc3_units_critic'],
        decay_rate=training_config['decay_rate'],
        explore_stop=training_config['explore_stop'],
        tensordir=f'{output_dir}\\runs\\',
        checkpoint_freq=10,
        checkpoint_path=f'{output_dir}\\weights\\',
        resume_from=resume_from,
        learn_freq=training_config['learn_freq'],
        plots=False,
        progress="tqdm",
        features_per_asset=features_per_asset,
        encoding_size=training_config['encoding_size'],
        clip_grad_norm=training_config['clip_grad_norm'],
        early_stop_patience=training_config['early_stop_patience']
    )
    
    # 10. Save results and visualizations
    save_results(results, output_dir, valid_tickers, enhancement_config)
    plot_enhanced_performance(results, output_dir, valid_tickers, enhancement_config)
    
    print(f"Enhanced training completed!")
    print(f"Trained models have been saved in: {output_dir}\\weights\\")
    print(f"TensorBoard logs have been saved in: {output_dir}\\runs\\")

    # 11. Setup test environment with the same enhancements
    print("\nPreparing test environment...")
    test_env = create_enhanced_environment(
        args, valid_tickers, aligned_dfs_test, 
        norm_params_paths, len(next(iter(aligned_dfs_test.values()))), 
        f'{output_dir}\\test',
        config=profile_config,  # Passa l'intero oggetto di configurazione
        calendar=calendar
    )
    
    # 12. Load the best model for evaluation
    model_files = [f for f in os.listdir(f'{output_dir}\\weights\\') 
                  if f.startswith('portfolio_actor_') and f.endswith('.pth')]
    
    if model_files:
        # Filter only numeric models, excluding 'initial'
        numeric_models = [f for f in model_files if f.split('_')[-1].split('.')[0].isdigit()]
        
        if numeric_models:
            last_model = sorted(numeric_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        else:
            # If no numeric models, use initial or first available
            last_model = next((f for f in model_files if 'initial' in f), model_files[0] if model_files else None)
        
        if last_model:
            last_critic = last_model.replace('actor', 'critic')
            
            print(f"Loading best model: {last_model}")
            agent.load_models(
                actor_path=f'{output_dir}\\weights\\{last_model}',
                critic_path=f'{output_dir}\\weights\\{last_critic}' if os.path.exists(f'{output_dir}\\weights\\{last_critic}') else None
            )
            
            # 13. Run evaluation on test data
            print("Running evaluation on test dataset...")
            
            # If we're using backtesting framework
            if args.use_backtest_framework:
                print("Using backtesting framework for walk-forward evaluation...")
                # Create a copy of test data in the format expected by backtester
                test_periods = {'full_test': {
                    'dates': next(iter(aligned_dfs_test.values()))['date'].values,
                    **{f'data_{ticker}': df.values for ticker, df in aligned_dfs_test.items()}
                }}
                
                # Initialize backtesting framework
                backtest = BacktestFramework(
                    env=test_env, 
                    agent=agent, 
                    datasets=test_periods,
                    test_periods=['full_test']
                )
                
                # Run walk-forward validation
                results = backtest.walk_forward_validation(
                    window_size=60,  # 60-day windows
                    step_size=20     # Step forward 20 days at a time
                )
                
                # Extract and analyze results
                print("\n=== Walk-Forward Validation Results ===")
                all_returns = []
                all_sharpes = []
                all_drawdowns = []
                
                for period, period_results in results.items():
                    period_returns = [r['total_return'] for r in period_results]
                    period_sharpes = [r['sharpe_ratio'] for r in period_results]
                    period_drawdowns = [r['max_drawdown'] for r in period_results]
                    
                    all_returns.extend(period_returns)
                    all_sharpes.extend(period_sharpes)
                    all_drawdowns.extend(period_drawdowns)
                    
                    print(f"\nPeriod: {period}")
                    print(f"Mean Return: {np.mean(period_returns):.2f}%")
                    print(f"Mean Sharpe: {np.mean(period_sharpes):.2f}")
                    print(f"Mean Drawdown: {np.mean(period_drawdowns):.2f}%")
                
                print("\nOverall Performance:")
                print(f"Mean Return: {np.mean(all_returns):.2f}%")
                print(f"Mean Sharpe: {np.mean(all_sharpes):.2f}")
                print(f"Mean Drawdown: {np.mean(all_drawdowns):.2f}%")
                
                # Save detailed backtest results
                backtest_results_file = f'{output_dir}\\test\\backtest_results.json'
                with open(backtest_results_file, 'w') as f:
                    import json
                    # Convert numpy arrays to lists for serialization
                    serializable_results = {
                        period: [{k: v if not isinstance(v, np.ndarray) else v.tolist() 
                                for k, v in r.items()} 
                                for r in period_results]
                        for period, period_results in results.items()
                    }
                    json.dump(serializable_results, f, indent=4)
                
                print(f"Detailed backtest results saved to: {backtest_results_file}")
                
            else:
                # Standard evaluation without backtesting framework
                test_env.reset()
                state = test_env.get_state()
                done = test_env.done
                
                # Nella parte di test, dopo aver caricato il modello
                print("\nInizio valutazione dettagliata sul dataset di test...")
                test_env.reset()
                state = test_env.get_state()
                done = test_env.done

                # Crea strutture dati per il logging
                action_log = []
                position_log = []
                portfolio_value_log = []
                rewards_log = []

                step_counter = 0
                significant_trade_counter = 0

                while not done:
                    with torch.no_grad():
                        actions = agent.act(state, noise=False)
                    
                    # Log prima dell'azione
                    if step_counter % 20 == 0 or np.any(np.abs(actions) > 0.1):  # Log ogni 20 step o per azioni significative
                        print(f"\nStep {step_counter}:")
                        print(f"Posizioni correnti: {test_env.positions}")
                        print(f"Azioni: {actions}")
                        print(f"Portfolio value: ${test_env.get_portfolio_value():.2f}")
                    
                    reward = test_env.step(actions)
                    state = test_env.get_state()
                    done = test_env.done
                    
                    # Registra se l'operazione era significativa
                    if np.any(np.abs(actions) > 0.001):
                        significant_trade_counter += 1
                    
                    # Salva i log
                    action_log.append(actions)
                    position_log.append(test_env.positions.copy())
                    portfolio_value_log.append(test_env.get_portfolio_value())
                    rewards_log.append(reward)
                    
                    step_counter += 1

                # Stampa riepilogo delle operazioni
                print("\n===== RIEPILOGO STRATEGIE =====")
                print(f"Numero totale di step: {step_counter}")
                print(f"Numero di operazioni significative: {significant_trade_counter}")
                print(f"Percentuale di step con trading: {significant_trade_counter/step_counter*100:.2f}%")

                # Analisi delle posizioni
                avg_positions = np.mean(position_log, axis=0)
                max_positions = np.max(position_log, axis=0)
                min_positions = np.min(position_log, axis=0)

                print("\n===== ANALISI POSIZIONI =====")
                print("Asset\tMedia\tMax\tMin")
                for i, ticker in enumerate(test_env.tickers):
                    print(f"{ticker}\t{avg_positions[i]:.4f}\t{max_positions[i]:.4f}\t{min_positions[i]:.4f}")

                # Calcola la correlazione tra le posizioni
                if len(position_log) > 5:
                    position_data = np.array(position_log)
                    print("\n===== CORRELAZIONI TRA POSIZIONI =====")
                    corr_matrix = np.corrcoef(position_data.T)
                    for i in range(len(test_env.tickers)):
                        for j in range(i+1, len(test_env.tickers)):
                            if not np.isnan(corr_matrix[i, j]):
                                print(f"Correlazione {test_env.tickers[i]}-{test_env.tickers[j]}: {corr_matrix[i, j]:.4f}")

                # Analizza se l'agente sta effettivamente facendo trading o rimanendo inattivo
                print("\n===== ATTIVITÀ DI TRADING =====")
                variance_positions = np.var(position_log, axis=0)
                print(f"Varianza delle posizioni: {variance_positions}")
                if np.all(variance_positions < 0.01):
                    print("ATTENZIONE: L'agente è praticamente inattivo - varianza di posizione troppo bassa")
                    
                # Analisi di diversificazione
                print("\n===== METRICHE DI DIVERSIFICAZIONE =====")
                avg_abs_positions = np.mean(np.abs(position_log), axis=0)
                if np.all(avg_abs_positions < 0.05):
                    print("ATTENZIONE: Agente troppo conservativo - posizioni medie troppo piccole")
                elif np.sum(avg_abs_positions > 0.5) == 1:
                    print("ATTENZIONE: Scarsa diversificazione - una posizione domina")
                else:
                    concentration = np.sum(avg_abs_positions**2) / (np.sum(avg_abs_positions)**2)
                    print(f"Indice di concentrazione: {concentration:.4f} (valori più bassi = maggiore diversificazione)")

                # Stampa risultati finali
                metrics = test_env.get_real_portfolio_metrics()
                print("\n===== RISULTATI FINALI =====")
                print(f"Rendimento totale: {metrics['total_return']:.2f}%")
                print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
                print(f"Valore finale portafoglio: ${metrics['final_portfolio_value']:.2f}")
                
                # If using market regimes, analyze performance by regime
                if args.use_market_regimes and regime_detector is not None:
                    ticker = test_env.tickers[0]
                    price_history = np.array(test_env.price_history[ticker])
                    regimes = []
                    
                    # Detect regimes at each step
                    for i in range(regime_detector.window_size, len(price_history)):
                        window = price_history[i-regime_detector.window_size:i]
                        regime = regime_detector.detect_regime(window)
                        regimes.append(regime)
                    
                    # Group rewards by regime
                    regime_rewards = {}
                    for regime, reward in zip(regimes, rewards_log[-len(regimes):]):
                        if regime not in regime_rewards:
                            regime_rewards[regime] = []
                        regime_rewards[regime].append(reward)
                    
                    print("\nPerformance by Market Regime:")
                    for regime, rewards in regime_rewards.items():
                        print(f"Regime {regime}:")
                        print(f"  Mean Reward: {np.mean(rewards):.4f}")
                        print(f"  Std Reward: {np.std(rewards):.4f}")
                        print(f"  Days in Regime: {len(rewards)}")
        else:
            print("No model found for evaluation.")
    else:
        print("No model found for evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Portfolio Training with Advanced DRL Features')
    
     # Gruppi di argomenti per organizzare meglio l'interfaccia
    base_group = parser.add_argument_group('Parametri base')
    enh_group = parser.add_argument_group('Enhancement features')

    # Parametri base
    base_group.add_argument('--output_dir', type=str, default='results/enhanced_portfolio', 
                          help='Directory for saving models and results')
    base_group.add_argument('--resume', type=str, default=None,
                          help='Path to checkpoint for resuming training')
    base_group.add_argument('--episodes', type=int, default=200,
                          help='Number of training episodes')
    base_group.add_argument('--test_ratio', type=float, default=0.2,
                          help='Ratio of data to use for testing')
    base_group.add_argument('--commission_rate', type=float, default=0.0025,
                          help='Trading commission rate')
    base_group.add_argument('--free_trades', type=int, default=10,
                          help='Number of free trades per month')
    base_group.add_argument('--learning_setup', type=str, choices=['full', 'fast'], default='full',
                          help='Choose between full or fast training setup')
    
    # Enhancement features
    enh_group.add_argument('--use_adaptive_exploration', action='store_true',
                         help='Use adaptive exploration strategy')
    enh_group.add_argument('--use_market_regimes', action='store_true',
                         help='Use market regime detection')
    enh_group.add_argument('--use_distributional', action='store_true',
                         help='Use distributional critic')
    enh_group.add_argument('--use_meta_learning', action='store_true',
                         help='Use meta-learning for faster adaptation')
    enh_group.add_argument('--use_calendar', action='store_true',
                         help='Use financial calendar data')
    enh_group.add_argument('--use_backtest_framework', action='store_true',
                         help='Use backtesting framework for evaluation')
    enh_group.add_argument('--calendar_path', type=str, default='calendar.csv',
                         help='Path to financial events calendar CSV')
    
     # Aggiungi gli argomenti per i profili di trading
    add_profile_arguments(parser)

    args = parser.parse_args()
    
    main(args)