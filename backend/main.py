from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model_lstm import (
    SEQUENCE_LENGTH,
    load_bundle,
    one_step_on_history,
    predict_future,
    save_bundle,
    train_lstm,
)

MODEL_STORE = os.path.join(os.path.dirname(__file__), "models_store")
os.makedirs(MODEL_STORE, exist_ok=True)

app = FastAPI(title="Stock LSTM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_closes(ticker: str, period: str) -> pd.Series:
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if df.empty or "Close" not in df.columns:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty or "Close" not in df.columns:
        raise ValueError(
            f"No price data for symbol {ticker!r}. "
            "Check the ticker and your network; if this persists, run: "
            "pip install -U 'yfinance>=1.3.0' and restart the server."
        )
    s = df["Close"].dropna()
    if len(s) < SEQUENCE_LENGTH + 20:
        raise ValueError(
            f"Not enough rows for {ticker} (need ~{SEQUENCE_LENGTH + 20}+). "
            "Try a longer period."
        )
    return s


class TrainRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)
    period: str = Field(default="2y", description="yfinance period, e.g. 1y, 2y, 5y, max")
    epochs: int = Field(default=30, ge=5, le=150)
    forecast_days: int = Field(default=10, ge=1, le=30)


class TrainResponse(BaseModel):
    ticker: str
    rows: int
    metrics: dict
    forecast: list[dict]
    history: list[dict]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/train", response_model=TrainResponse)
def train(req: TrainRequest):
    ticker = req.ticker.strip().upper()
    try:
        closes = fetch_closes(ticker, req.period)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        model, scaler, metrics = train_lstm(closes, epochs=req.epochs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}") from e

    fitted_tail = one_step_on_history(model, scaler, closes)

    horizon = predict_future(model, scaler, closes, days=req.forecast_days)

    meta = {
        "period": req.period,
        "rows": len(closes),
        "last_close": float(closes.iloc[-1]),
        "fitted_tail": float(fitted_tail),
    }
    save_bundle(MODEL_STORE, ticker, model, scaler, meta)

    last_ts = closes.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    forecast_out = []
    for i, price in enumerate(horizon, start=1):
        dt = last_ts + timedelta(days=i)
        forecast_out.append({"day_offset": i, "date": dt.date().isoformat(), "price": float(price)})

    history = [
        {"date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx), "close": float(v)}
        for idx, v in closes.iloc[-120:].items()
    ]

    return TrainResponse(
        ticker=ticker,
        rows=len(closes),
        metrics=metrics,
        forecast=forecast_out,
        history=history,
    )


class PredictRequest(BaseModel):
    ticker: str
    days: int = Field(default=10, ge=1, le=30)
    period: str = Field(default="2y")


@app.post("/api/predict")
def predict(req: PredictRequest):
    ticker = req.ticker.strip().upper()
    loaded = load_bundle(MODEL_STORE, ticker)
    model, scaler, meta = loaded
    if model is None or scaler is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {ticker}. Run train first.",
        )

    try:
        closes = fetch_closes(ticker, req.period)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    horizon = predict_future(model, scaler, closes, days=req.days)
    last_ts = closes.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    forecast_out = []
    for i, price in enumerate(horizon, start=1):
        dt = last_ts + timedelta(days=i)
        forecast_out.append({"day_offset": i, "date": dt.date().isoformat(), "price": float(price)})

    history = [
        {"date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx), "close": float(v)}
        for idx, v in closes.iloc[-120:].items()
    ]

    return {
        "ticker": ticker,
        "meta": meta,
        "forecast": forecast_out,
        "history": history,
    }


_frontend = os.path.join(os.path.dirname(__file__), "..", "frontend")
_frontend_index = os.path.join(_frontend, "index.html")


@app.get("/")
def serve_ui():
    if not os.path.isfile(_frontend_index):
        return {"error": "frontend/index.html missing"}
    return FileResponse(_frontend_index)
