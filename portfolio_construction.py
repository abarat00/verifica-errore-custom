# New file: portfolio_construction.py
import numpy as np
import scipy

class HybridPortfolioConstructor:
    def __init__(self, drl_agent, risk_aversion=2.0):
        self.drl_agent = drl_agent
        self.risk_aversion = risk_aversion
        
    def optimize_allocation(self, state, expected_returns, covariance_matrix):
        """
        Combine DRL recommendations with mean-variance optimization.
        """
        # Get DRL agent's recommendation
        drl_actions = self.drl_agent.act(state, noise=False)
        
        # Perform mean-variance optimization
        n_assets = len(expected_returns)
        
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            return -(portfolio_return - self.risk_aversion * portfolio_risk)
        
        # Constraints: sum of weights = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds for each weight
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess - use the DRL actions as starting point
        # Convert actions to valid weights between 0 and 1
        initial_guess = np.abs(drl_actions) / (np.sum(np.abs(drl_actions)) + 1e-8)
        
        # Optimize
        result = scipy.optimize.minimize(
            objective, initial_guess, method='SLSQP', 
            bounds=bounds, constraints=constraints
        )
        
        # Blend DRL and MVO weights
        alpha = 0.5  # Blend factor - equal weight to both approaches
        blended_weights = alpha * result['x'] + (1 - alpha) * initial_guess
        
        return blended_weights