# strading-signals-TSLA-using-Elon-s-tweets

## Introduction and Goals

The goal of this project is to explore new ways to identify trading signals.  
The analysis focuses on **Tesla stock**, using a model that predicts **long (buy)** and **short (sell)** signals.

The model uses both **financial data** and **sentiment data extracted from Elon Musk’s tweets**, given Musk’s strong connection with Tesla.  
The input data includes:
- Closing prices  
- Trading volumes  
- Log returns  
- Sentiment derived from tweets  

The main objective is to **predict the market direction of the next trading day**, specifically the **sign of the log return**, using information from past days.

Initially, a **binary classification model** (long vs short) was tested, but it performed poorly.  
The approach was then extended to a **three-class classification problem**:
- Long (buy)
- Short (sell)
- Flat (do nothing)

## Evaluation Metrics

Model performance is evaluated using standard financial metrics:

- **Total Return**  
  \[
  (S_T - S_0) / S_0
  \]

- **Sharpe Ratio**  
  \[
  (average P&L - r) / std(P&L)
  \]

- **Win Rate**  
  \[
  number of winning trades / total trades
  \]

- **Max Drawdown**  
  Minimum drawdown over time, where drawdown is defined as:
  \[
  (S_t - max_{s < t} S_s) / max_{s < t} S_s
  \]

Where:
- `S_T` is the stock price at the end of the investment period  
- `S_0` is the initial stock price  
- `r` is the risk-free rate  
- Profit and Loss (P&L) refers to trading gains and losses  

Assumptions:
- Risk-free rate is set to **0**
- Trading fees are ignored

The dataset spans from **2010-06-04** to **2023-06-29**.

## Sentiment Analysis

Sentiment analysis is applied to Elon Musk’s tweets by mapping each tweet to a probability distribution over:
- Negative sentiment  
- Neutral sentiment  
- Positive sentiment  

These probabilities sum to 1.

The **Gemini API** is used for sentiment extraction and was found to outperform FinBERT and fine-tuned BERT models on tweets.

Tweets are grouped by trading day, and the daily sentiment score is computed as:

sentiment = p_positive - p_negative

This value lies in the range `[-1, 1]`.

For each trading day, the final dataset includes:
- Average daily sentiment score
- Number of tweets posted that day

## Financial Data and Data Handling

The financial features used are:
- Log returns  
- Trading volume  
- Sign or class of log returns  

### Trading Day Definition and Data Alignment

Tesla trades on NASDAQ, with market close at **21:00 UTC**, and no trading on weekends or holidays.  
Tweets can be posted at any time, so careful alignment is required to avoid data leakage.

A **trading day** is defined as the period between two consecutive market closes:
- Tweets posted **before 21:00 UTC** affect the same day’s close
- Tweets posted after close, or during weekends/holidays, affect the **next trading close**

Within each trading day:
- Sentiments are averaged
- Tweets are counted

### Dataset Preparation

For neural network training:
- **Labels**: sign or class of the **next day’s log return**
- **Features**: rolling window of **128 (or 64) past trading days**, including:
  - Trading volume  
  - Log returns  
  - Sign or class of log returns  
  - Average sentiment score  
  - Tweet count  

The dataset is split into **training, validation, and test sets**, preserving time order (no shuffling).

## Model Choice

The classification model is a neural network with the following architecture:
- Two **Bidirectional LSTM** layers:
  - First layer: 64 units  
  - Second layer: 32 units  
- One **Dense layer** with 16 units and ReLU activation  
- Output layer:
  - Binary classification: 1 neuron with sigmoid activation  
  - Three-class classification: 3 neurons with softmax activation  

### Binary vs Three-Class Classification

The binary classifier performed poorly, consistently predicting the **long** class with probabilities between 51% and 57%.  
Adjusting thresholds to maximize the Sharpe Ratio did not improve performance, indicating that predictions were dominated by noise.

The three-class formulation was motivated by the idea that:
- Small price movements are often not worth trading
- Large movements matter more for profitability
- Staying flat during small fluctuations can reduce unnecessary risk

To define the three classes, **percentile thresholds** on training log returns were tested:
- (32, 68)
- (25, 75)
- (20, 80)

Classification rules:
- Below lower percentile → **Short**
- Above upper percentile → **Long**
- Between percentiles → **Flat**

## Results

The best performance on the test set was achieved using:
- **20–80 percentile thresholds**
- An **only-long strategy** (short signals ignored; positions are either long or flat)

Test set results over **288 trading days**:
- **Total Return**: 0.4186  
- **Sharpe Ratio**: 1.6956  
- **Max Drawdown**: -0.0632  
- **Win Rate**: 0.6842  

Number of trades: **19**

Despite the low number of trades, accuracy was high.  
The combination of strong returns, high Sharpe Ratio, and low drawdown indicates good risk control.  
The test period was predominantly bearish.

## Future Developments

Planned improvements include:
- Further tuning of network architecture and hyperparameters  
- Optimization of percentile thresholds for class separation  
- Reformulating the problem as a **regression task**, directly optimizing financial metrics (e.g. Sharpe Ratio) through custom loss functions  

