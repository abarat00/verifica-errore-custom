# New file: backtesting.py
class BacktestFramework:
    def __init__(self, env, agent, datasets, test_periods):
        self.env = env
        self.agent = agent
        self.datasets = datasets  # Dictionary of time periods and their data
        self.test_periods = test_periods  # List of period names to test
    
    def walk_forward_validation(self, window_size=60, step_size=20):
        """Perform walk-forward validation."""
        results = {}
        
        for period_name in self.test_periods:
            period_data = self.datasets[period_name]
            
            # Walk-forward windows
            windows = []
            for start in range(0, len(period_data['dates']) - window_size, step_size):
                end = start + window_size
                windows.append((start, end))
            
            period_results = []
            for start, end in windows:
                # Create a subset of data for this window
                window_data = self.prepare_window_data(period_data, start, end)
                
                # Update environment with window data
                self.env.update_data(window_data)
                
                # Test agent
                metrics = self.test_agent_performance()
                metrics['window'] = (period_data['dates'][start], period_data['dates'][end-1])
                period_results.append(metrics)
            
            results[period_name] = period_results
        
        return results
        
    def prepare_window_data(self, period_data, start, end):
        # Extract relevant slice of data for testing window
        window_data = {}
        for key, value in period_data.items():
            if key != 'dates':
                window_data[key] = value[start:end]
        return window_data
    
    def test_agent_performance(self):
        # Reset environment and test agent
        self.env.reset()
        state = self.env.get_state()
        done = False
        rewards = []
        
        while not done:
            action = self.agent.act(state, noise=False)
            reward = self.env.step(action)
            next_state = self.env.get_state()
            state = next_state
            done = self.env.done
            rewards.append(reward)
        
        return self.env.get_real_portfolio_metrics()