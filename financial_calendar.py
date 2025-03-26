import pandas as pd
from datetime import datetime, timedelta

class FinancialCalendar:
    def __init__(self, data_path=None):
        self.events = {}
        self.importance_levels = {'N': 0, 'L': 1, 'M': 2, 'H': 3}
        if data_path:
            self.load_calendar(data_path)
    
    def load_calendar(self, data_path):
        """Load financial events from simplified CSV (date, importance, description)."""
        events_loaded = 0
        
        try:
            # Leggi il file come testo
            with open(data_path, 'r') as file:
                lines = file.readlines()
            
            # Salta l'intestazione se presente
            if lines and ('date' in lines[0].lower() or 'importance' in lines[0].lower()):
                lines = lines[1:]
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Formato semplificato: data,importanza,descrizione
                try:
                    parts = line.split(',', 2)  # Split solo sulle prime 2 virgole
                    
                    if len(parts) < 2:
                        print(f"Skipping line with insufficient parts: {line}")
                        continue
                    
                    date_str = parts[0].strip()
                    importance = parts[1].strip()
                    description = parts[2].strip() if len(parts) > 2 else ""
                    
                    # Parse della data
                    try:
                        date = pd.to_datetime(date_str).date()
                    except Exception as e:
                        print(f"Cannot parse date '{date_str}': {e}")
                        continue
                    
                    # Crea e salva l'evento
                    event = {
                        'type': 'general',
                        'importance': importance,
                        'description': description
                    }
                    
                    if date not in self.events:
                        self.events[date] = []
                    
                    self.events[date].append(event)
                    events_loaded += 1
                
                except Exception as e:
                    print(f"Error processing line: {line}")
                    print(f"Error details: {e}")
                    continue
            
            print(f"Successfully loaded {events_loaded} events from calendar")
        
        except Exception as e:
            print(f"Error reading calendar file: {e}")
    
    def get_upcoming_events(self, current_date, lookahead=7, tickers=None):
        """
        Get important events in the next N days.
        
        Parameters:
        - current_date: Date to look from
        - lookahead: Number of days to look ahead
        - tickers: Ignored in this simplified version
        
        Returns:
        - List of upcoming events
        """
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date).date()
        elif isinstance(current_date, pd.Timestamp):
            current_date = current_date.date()
            
        upcoming = []
        
        for i in range(lookahead):
            check_date = current_date + timedelta(days=i)
            if check_date in self.events:
                for event in self.events[check_date]:
                    upcoming.append({
                        'days_ahead': i,
                        'date': check_date,
                        'type': event.get('type', 'general'),
                        'importance': event['importance'],
                        'importance_score': self.importance_levels.get(event['importance'], 0),
                        'description': event.get('description', '')
                    })
        
        return upcoming
    
    def get_event_features(self, current_date, tickers, lookahead=7):
        """
        Generate features for the ML model based on upcoming events.
        Returns a vector of features.
        """
        upcoming = self.get_upcoming_events(current_date, lookahead)
        
        # Initialize features
        features = {
            'high_importance_count': 0,  # H
            'medium_importance_count': 0,  # M
            'low_importance_count': 0,  # L
            'days_to_next_high': lookahead + 1,  # Default if none found
            'days_to_next_medium': lookahead + 1,
            'days_to_next_any': lookahead + 1,
            'event_importance_weighted': 0  # Score ponderato
        }
        
        # Calcola i conteggi e le distanze
        for event in upcoming:
            # Conteggi per importanza
            importance = event['importance']
            if importance == 'H':
                features['high_importance_count'] += 1
            elif importance == 'M':
                features['medium_importance_count'] += 1
            elif importance == 'L':
                features['low_importance_count'] += 1
            # Ignoriamo gli eventi 'N' (festivities)
            
            # Giorni al prossimo evento
            days_ahead = event['days_ahead']
            if importance == 'H' and days_ahead < features['days_to_next_high']:
                features['days_to_next_high'] = days_ahead
            if importance == 'M' and days_ahead < features['days_to_next_medium']:
                features['days_to_next_medium'] = days_ahead
            if importance != 'N' and days_ahead < features['days_to_next_any']:
                features['days_to_next_any'] = days_ahead
                
            # Score ponderato per importanza e vicinanza
            importance_score = self.importance_levels.get(importance, 0)
            proximity_weight = 1.0 / (days_ahead + 1)  # Più vicino = peso maggiore
            features['event_importance_weighted'] += importance_score * proximity_weight
        
        # Normalizza gli score
        normalization_factor = max(lookahead, 1)
        features['event_importance_weighted'] /= normalization_factor
            
        return features