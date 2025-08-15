import numpy as np

def generate_probabilities(predicted_prices, current_price):
    """Generates buy/sell probabilities based on the average predicted price change."""
    price_changes = predicted_prices - current_price
    avg_change = np.mean(price_changes)
    sensitivity = 2000
    buy_logit = max(0, avg_change) * sensitivity
    sell_logit = max(0, -avg_change) * sensitivity
    exp_buy = np.exp(buy_logit)
    exp_sell = np.exp(sell_logit)
    buy_prob = exp_buy / (1 + exp_buy + exp_sell)
    sell_prob = exp_sell / (1 + exp_buy + exp_sell)
    return buy_prob, sell_prob

def generate_confidence(predicted_prices, atr):
    """
    Generates a confidence score based on prediction spread and magnitude of change.
    """
    spread = np.std(predicted_prices)
    normalized_spread = 1 - min(1, spread / (atr + 1e-9))
    avg_predicted_change = np.abs(np.mean(predicted_prices) - predicted_prices[0])
    normalized_magnitude = min(1, avg_predicted_change / (atr + 1e-9))
    confidence = (normalized_spread * 0.4) + (normalized_magnitude * 0.6)
    return max(0.0, min(1.0, confidence))