# New file: market_regime.py
import numpy as np
from hmmlearn import hmm

class MarketRegimeDetector:
    def __init__(self, window_size=60, n_regimes=3):
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.hmm_model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full")
        
    def detect_regime(self, price_history):
        """Detect the current market regime using HMM."""
        if len(price_history) < self.window_size:
            return 0  # Default regime
            
        # Extract features from recent price history
        returns = np.diff(price_history[-self.window_size:]) / price_history[-self.window_size:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        # Using returns and volatility as features
        X = np.column_stack([returns, np.abs(returns)])
        
        if len(X) < self.window_size - 1:
            return 0
            
        # Fit the model if we have enough data
        self.hmm_model.fit(X)
        
        # Predict the current regime
        current_regime = self.hmm_model.predict(X)[-1]
        return current_regime