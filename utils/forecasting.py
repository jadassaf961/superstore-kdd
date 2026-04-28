"""
Lightweight time-series forecasting — NO Prophet (per user request).

Two models:
  • Seasonal-naive baseline — last year's same-month value
  • SARIMA via statsmodels — captures trend + yearly seasonality

Both fit in <1s on the monthly-aggregated Superstore series (48 months).
"""
import numpy as np
import pandas as pd
import warnings


def aggregate_monthly(df: pd.DataFrame, value_col: str = "Total Revenue") -> pd.Series:
    """Aggregate to monthly totals, indexed by month-start."""
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    s = (df.dropna(subset=["Order Date"])
           .set_index("Order Date")[value_col]
           .resample("MS").sum()
           .asfreq("MS")
           .fillna(0))
    return s


# ══════════════════════════════════════════════════════════════════════════
# Seasonal-naive baseline: forecast = value 12 months ago
# ══════════════════════════════════════════════════════════════════════════
def seasonal_naive_forecast(series: pd.Series, horizon: int = 6,
                             season: int = 12) -> pd.DataFrame:
    """Baseline: ŷ_{t+h} = y_{t+h-season}. Returns DataFrame with forecast + bands."""
    last_date = series.index[-1]
    future_idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                periods=horizon, freq="MS")
    forecasts = []
    for i, dt in enumerate(future_idx, start=1):
        # Look back `season` months from forecast point
        ref_date = dt - pd.DateOffset(months=season)
        if ref_date in series.index:
            forecasts.append(series.loc[ref_date])
        else:
            forecasts.append(series.iloc[-season] if len(series) >= season else series.mean())

    # Empirical residual std for confidence bands (last 12 months error vs. seasonal naive)
    resid = []
    for i in range(season, len(series)):
        resid.append(series.iloc[i] - series.iloc[i - season])
    sigma = np.std(resid) if resid else series.std()

    fc = pd.DataFrame({
        "ds": future_idx,
        "yhat": forecasts,
        "yhat_lower": np.array(forecasts) - 1.96 * sigma,
        "yhat_upper": np.array(forecasts) + 1.96 * sigma,
        "model": "Seasonal Naive",
    })
    return fc


# ══════════════════════════════════════════════════════════════════════════
# SARIMA via statsmodels
# ══════════════════════════════════════════════════════════════════════════
def sarima_forecast(series: pd.Series, horizon: int = 6,
                     order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) -> tuple:
    """
    Fit SARIMA and return (forecast_df, fitted_values_series).
    Forecast df includes ds / yhat / yhat_lower / yhat_upper / model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=horizon)
        mean = fc.predicted_mean
        conf = fc.conf_int(alpha=0.05)

    out = pd.DataFrame({
        "ds": mean.index,
        "yhat": mean.values,
        "yhat_lower": conf.iloc[:, 0].values,
        "yhat_upper": conf.iloc[:, 1].values,
        "model": "SARIMA",
    })
    fitted = pd.Series(res.fittedvalues, index=series.index)
    return out, fitted, res


# ══════════════════════════════════════════════════════════════════════════
# Backtesting — last `horizon` months held out
# ══════════════════════════════════════════════════════════════════════════
def backtest(series: pd.Series, horizon: int = 6, model: str = "sarima") -> dict:
    """Hold out last `horizon` months, train on the rest, score."""
    if len(series) <= horizon + 12:
        return {"error": "Not enough history for backtesting"}
    train = series.iloc[:-horizon]
    test  = series.iloc[-horizon:]

    if model == "sarima":
        fc_df, _, _ = sarima_forecast(train, horizon=horizon)
        pred = fc_df["yhat"].values
    else:
        fc_df = seasonal_naive_forecast(train, horizon=horizon)
        pred = fc_df["yhat"].values

    actual = test.values
    mae  = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / np.where(actual == 0, 1, actual))) * 100)

    return {
        "model": model,
        "horizon": horizon,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "train_end": str(train.index[-1].date()),
        "test_start": str(test.index[0].date()),
        "test_actual": actual.tolist(),
        "test_pred":   pred.tolist(),
        "test_dates":  [str(d.date()) for d in test.index],
    }
