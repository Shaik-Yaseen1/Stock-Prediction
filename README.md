# LSTM Stock & Crypto Forecast

End-to-end web app that downloads historical market data, trains a stacked **LSTM** on daily adjusted closes, and projects a short price horizon. Built for **learning and prototyping** — not financial advice or live trading.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![JAX](https://img.shields.io/badge/JAX-Equinox-orange)

---

## Features

- **Train & forecast** — fetch data from Yahoo Finance, fit an LSTM, plot history vs projected path
- **Quick predict** — reload a saved model from disk without retraining
- **Stocks & crypto** — e.g. `AAPL`, `MSFT`, `BTC-USD`, `ETH-USD`
- **REST API** — FastAPI with OpenAPI docs at `/docs`
- **Web UI** — single-page dashboard with Chart.js

---

## Tech stack

| Layer | Technologies |
|-------|----------------|
| **API** | Python, FastAPI, Uvicorn, Pydantic |
| **ML** | JAX, Equinox, Optax, scikit-learn |
| **Data** | pandas, yfinance (Yahoo Finance) |
| **Frontend** | HTML, vanilla JavaScript, Tailwind CSS, Chart.js |

---

## Project structure

```
Stock Prediction/
├── backend/
│   ├── main.py           # FastAPI routes & data fetch
│   ├── model_lstm.py     # Stacked LSTM (60-day sequences)
│   ├── requirements.txt
│   ├── runtime.txt       # Python version for Render
│   └── models_store/     # Saved models (gitignored)
├── frontend/
│   └── index.html        # Dashboard UI
├── run.sh                # Local dev server
├── render.yaml           # Render.com deploy blueprint
└── README.md
```

---

## Prerequisites

- **Python 3.13+**
- Internet access (Yahoo Finance via `yfinance`)

---

## Run locally

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd "Stock Prediction"

python3.13 -m venv .venv313
source .venv313/bin/activate   # Windows: .venv313\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Start the server

```bash
chmod +x run.sh   # first time only
./run.sh
```

Or manually:

```bash
source .venv313/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
```

### 3. Open the app

- **UI:** http://127.0.0.1:8000  
- **API docs:** http://127.0.0.1:8000/docs  
- **Health:** http://127.0.0.1:8000/api/health  

---

## Using the UI

1. **Train model** — enter a ticker (e.g. `AAPL` or `BTC-USD`), choose history length, epochs, and forecast days, then click **Train & forecast**. First run may take 1–2 minutes.
2. **Chart** — solid line = recent closes; dashed line = LSTM projection.
3. **Quick predict** — after training, use the same ticker to forecast again without retraining (uses weights in `backend/models_store/`).

### Ticker examples

| Symbol | Asset |
|--------|--------|
| `AAPL` | Apple stock |
| `MSFT` | Microsoft stock |
| `BTC-USD` | Bitcoin vs USD |
| `ETH-USD` | Ethereum vs USD |

Use Yahoo’s `-USD` suffix for crypto (not plain `BTC`).

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/train` | Download data, train LSTM, save model, return forecast |
| `POST` | `/api/predict` | Predict with saved model |

**Train** example:

```bash
curl -X POST http://127.0.0.1:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","period":"2y","epochs":30,"forecast_days":14}'
```

**Predict** example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","days":10,"period":"2y"}'
```

---

## Deploy on Render (GitHub)

1. Push this repo to **GitHub** (do not commit `.venv313/` — it is in `.gitignore`).
2. Sign in at [Render](https://dashboard.render.com).
3. **New → Blueprint** → connect the repo (uses `render.yaml`).
4. Wait for deploy, then open your `*.onrender.com` URL.

**Manual web service settings** (if not using Blueprint):

| Setting | Value |
|---------|--------|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

> **Note:** On Render’s free tier, services sleep when idle and long training requests may time out. Use fewer epochs for demos or run training locally.

---

## How the model works

- **Input:** last 60 days of scaled adjusted close prices (`SEQUENCE_LENGTH = 60`).
- **Architecture:** two stacked LSTM layers + dense head (JAX/Equinox).
- **Training:** Adam via Optax, early stopping on validation loss.
- **Forecast:** autoregressive multi-step prediction from the latest window.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `Expecting value: line 1 column 1` / no price data | Upgrade yfinance: `pip install -U 'yfinance>=1.3.0'` and restart the server |
| `No trained model for TICKER` | Run **Train & forecast** before **Quick predict** |
| Not enough rows | Use a longer history period (`5y` or `max`) |
| API shows offline in UI | Ensure `./run.sh` or uvicorn is running on port 8000 |

---

## Disclaimer

This project is for **education and experimentation** only. Forecasts are illustrative and must not be used as investment or trading advice. Past performance does not guarantee future results.

---

## License

MIT — use and modify freely for learning and portfolio purposes.
