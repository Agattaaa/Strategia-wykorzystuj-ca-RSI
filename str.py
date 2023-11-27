import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

# Function to calculate Annualized Return Compounded (ARC)
def calculate_arc(balance, initial_balance, trading_periods_per_year):
    if initial_balance >= balance:
        return 0  # Avoiding negative value in the power

    total_return = (balance / initial_balance) ** (1 / trading_periods_per_year) - 1
    return total_return

# Function to calculate Annualized Standard Deviation (ASD)
def calculate_asd(returns, trading_periods_per_year):
    annualized_volatility = np.std(returns) * np.sqrt(trading_periods_per_year)
    return annualized_volatility

# Function to calculate Max Drawdown (MDD) and Maximum Loss Duration (MLD)
def calculate_max_drawdown(returns):
    cum_returns = np.cumprod(1 + returns) - 1
    max_drawdown = np.min(cum_returns - np.maximum.accumulate(cum_returns))
    max_loss_duration = np.argmax(np.maximum.accumulate(cum_returns) - cum_returns)
    return max_drawdown, max_loss_duration

# Function to calculate Information Ratio
def calculate_information_ratio(returns, benchmark_returns):
    excess_returns = returns - benchmark_returns
    information_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return information_ratio


def calculate_position_changes(signals):
    total_trading_days = len(signals)
    position_changes = np.sum(np.abs(signals['Buy_Signal'].diff()) + np.abs(signals['Sell_Signal'].diff()))

    position_changes = (position_changes / total_trading_days) * 100
    return position_changes

#pobranie danych
data = yf.download('^SPX', start='1988-12-31', end='2023-10-30')

#RSI
def calculate_rsi(data, period=14):
    data = data.copy()

    close_delta = data['Close'].diff(1)
    gain = close_delta.where(close_delta > 0, 0)
    loss = -close_delta.where(close_delta < 0, 0)

    data['avg_gain'] = gain.rolling(window=period, min_periods=1).mean()
    data['avg_loss'] = loss.rolling(window=period, min_periods=1).mean()

    rs = data['avg_gain'] / data['avg_loss']
    data['RSI'] = 100 - (100 / (1 + rs))

    return data['RSI']

# wyliczenie RSI
data.loc[:, 'RSI'] = calculate_rsi(data)

# Generowanie sygnałów
def generate_signals(data, overbought_threshold=70, oversold_threshold=30, period=14):
    signals = pd.DataFrame(index=data.index)
    signals['Close'] = data['Close']

    # Buy signals
    signals['Buy_Signal'] = (data['RSI'] < oversold_threshold) & (data['RSI'].shift(1) >= oversold_threshold)

    # Sell signals
    signals['Sell_Signal'] = (data['RSI'] > overbought_threshold) & (data['RSI'].shift(1) <= overbought_threshold)

    return signals

# Symulacja strategii
def simulate_strategy(signals, initial_balance=100000, transaction_cost=0.0005):
    balance = initial_balance
    position = 0

    for index, row in signals.iterrows():
        if row['Buy_Signal']:

            position = balance / row['Close']
            balance = 0
            balance -= transaction_cost * position * row['Close']

        elif row['Sell_Signal']:
            # Sell all stocks
            balance += position * row['Close']
            position = 0
            balance -= transaction_cost * balance


    balance += position * signals['Close'].iloc[-1]

    return balance

# Function to plot results
def plot_results(data, signals):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Stock Price')
    plt.scatter(signals.index[signals['Buy_Signal']], signals['Close'][signals['Buy_Signal']], marker='^',
                color='g', label='Buy Signal')
    plt.scatter(signals.index[signals['Sell_Signal']], signals['Close'][signals['Sell_Signal']], marker='v',
                color='r', label='Sell Signal')
    plt.title('RSI-Based Trading Strategy')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Parametry do symulacji
training_period = 2 * 252  # 2 years (252 dni tradingowych w roku)
validation_period = 252
testing_period = 252
transaction_cost = 0.0005  # 0.05%
print(len(data))
# Walk-forward testing
for i in range(0, len(data) - training_period - validation_period - testing_period + 1, testing_period):
    train_data = data.iloc[i:i + training_period]
    val_data = data.iloc[i + training_period:i + training_period + validation_period]
    test_data = data.iloc[i + training_period + validation_period:i + training_period + validation_period + testing_period]

    # Calculate RSI for the training data
    train_data.loc[:, 'RSI'] = calculate_rsi(train_data)

    # Generate signals for the validation data
    val_signals = generate_signals(train_data)

    # Plot results for better visualization
    #plot_results(train_data, val_signals)

    # Simulate the strategy for the testing data
    final_balance = simulate_strategy(val_signals, initial_balance=100000, transaction_cost=transaction_cost)

    # Print the final balance for each testing period
    print(f"Testing Period {i // testing_period + 1} - Final Balance: ${final_balance}")


def optimize_parameters(train_data, validation_data):
    best_balance = 0
    best_params = None

    for period in range(10, 31, 5):
        for overbought_threshold in range(70, 91, 5):
            for oversold_threshold in range(10, 31, 5):
                signals = generate_signals(train_data, overbought_threshold, oversold_threshold, period)
                balance = simulate_strategy(signals)

                if balance > best_balance:
                    best_balance = balance
                    best_params = {'period': period, 'overbought_threshold': overbought_threshold,
                                   'oversold_threshold': oversold_threshold}

    return best_params

results_list = []
# Walk-forward testing
for i in range(0, len(data) - training_period - validation_period - testing_period + 1, testing_period):
    train_data = data.iloc[i:i + training_period]
    val_data = data.iloc[i + training_period:i + training_period + validation_period]
    test_data = data.iloc[
                i + training_period + validation_period:i + training_period + validation_period + testing_period]

    # Optimize parameters on the training and validation data
    best_params = optimize_parameters(train_data, val_data)

    # Calculate RSI for the training data with optimized parameters
    train_data.loc[:, 'RSI'] = calculate_rsi(train_data, period=best_params['period'])

    # Generate signals for the validation data with optimized parameters
    val_signals = generate_signals(train_data, overbought_threshold=best_params['overbought_threshold'],
                                   oversold_threshold=best_params['oversold_threshold'], period=best_params['period'])

    # Plot results for better visualization
    #plot_results(train_data, val_signals)

    # Simulate the strategy for the testing data
    final_balance = simulate_strategy(val_signals, initial_balance=100000, transaction_cost=transaction_cost)

    # Print the final balance for each testing period
    print(f"Testing Period {i // testing_period + 1} - Final Balance: ${final_balance} (Optimized Parameters: {best_params})")
    # Calculate and print additional metrics
    returns = val_signals['Close'].pct_change().dropna()
    trading_periods_per_year = 252  # Assuming 252 trading days in a year

    arc = calculate_arc(final_balance, initial_balance=100000, trading_periods_per_year=trading_periods_per_year)
    asd = calculate_asd(returns, trading_periods_per_year=trading_periods_per_year)
    max_drawdown, max_loss_duration = calculate_max_drawdown(returns)
    information_ratio = calculate_information_ratio(returns, benchmark_returns=0)  # Assuming no benchmark
    position_changes = calculate_position_changes(val_signals)

    # Append results to the list
    results_list.append({
        'Testing Period': i // testing_period + 1,
        'Final Balance': final_balance,
        'ARC': arc,
        'ASD': asd,
        'Max Drawdown': max_drawdown,
        'Max Loss Duration': max_loss_duration,
        'Information Ratio': information_ratio,
        'Position Changes': position_changes
    })

# Create a DataFrame from the list of results
results_table = pd.DataFrame(results_list)

# Print the final results table
print(results_table)
results_table.to_excel('results.xlsx', index=False)


