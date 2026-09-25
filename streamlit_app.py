import math
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Option Screener", layout="wide")

# =========================================================
# Funzioni di supporto (Black-Scholes, catena opzioni, metriche)
# =========================================================

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(spot, strike, dte, iv, r, q, right):
    if iv is None or iv <= 0 or dte <= 0 or spot <= 0 or strike <= 0:
        return None
    T = dte / 365
    try:
        d1 = (math.log(spot / strike) + (r - q + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return None
    if right == "C":
        return math.exp(-q * T) * norm_cdf(d1)
    else:
        return math.exp(-q * T) * (norm_cdf(d1) - 1)


def get_spot_price(ticker):
    try:
        fi = ticker.fast_info
        price = fi.get("lastPrice") or fi.get("last_price")
        if price:
            return float(price)
    except Exception:
        pass
    try:
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def pick_expiries(expirations, dte_min: int, dte_max: int):
    today = date.today()
    picked = []
    for e in expirations:
        exp_date = datetime.strptime(e, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte_min <= dte <= dte_max:
            picked.append((e, dte))
    return picked


def build_quotes(df, spot, dte, right, cfg, window_pct):
    if df is None or df.empty:
        return []
    lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
    df = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
    quotes = []
    for _, row in df.iterrows():
        bid = row.get("bid", 0.0) or 0.0
        ask = row.get("ask", 0.0) or 0.0
        iv = row.get("impliedVolatility", None)
        oi = row.get("openInterest", 0) or 0
        mid = (bid + ask) / 2 if bid and ask else (row.get("lastPrice", 0.0) or 0.0)
        if mid <= 0 or oi < cfg["min_open_interest"]:
            continue
        if bid > 0 and ask > 0 and (ask - bid) / mid > cfg["max_bid_ask_spread_pct"]:
            continue
        delta = bs_delta(spot, row["strike"], dte, iv, cfg["risk_free_rate"], cfg["dividend_yield"], right)
        if delta is None:
            continue
        quotes.append({
            "strike": float(row["strike"]), "bid": float(bid), "ask": float(ask), "mid": float(mid),
            "delta": abs(delta), "iv": float(iv) if iv is not None else None, "oi": int(oi),
        })
    return quotes


def annualized_return(premium, capital, dte):
    if capital <= 0 or dte <= 0:
        return 0.0
    return (premium / capital) * (365 / dte)


def closest_by_delta(quotes, target_lo, target_hi):
    in_range = [q for q in quotes if target_lo <= q["delta"] <= target_hi]
    if in_range:
        mid_target = (target_lo + target_hi) / 2
        return min(in_range, key=lambda q: abs(q["delta"] - mid_target))
    return None


def closest_by_strike(quotes, target_strike):
    if not quotes:
        return None
    return min(quotes, key=lambda q: abs(q["strike"] - target_strike))


# =========================================================
# Scanner per strategia (stessa logica testata nel notebook)
# =========================================================

def scan_covered_call(symbol, spot, expiry, dte, calls, cfg, results):
    calls_ok = [q for q in calls if q["strike"] > spot]
    pick = closest_by_delta(calls_ok, cfg["cc_delta_min"], cfg["cc_delta_max"])
    if not pick:
        return
    ann_ret = annualized_return(pick["mid"], spot, dte)
    if ann_ret < cfg["cc_min_annualized_return"]:
        return
    results.append({
        "strategia": "Covered Call", "simbolo": symbol, "scadenza": expiry, "dte": dte,
        "spot": round(spot, 2), "strike_short": pick["strike"], "premio": round(pick["mid"], 2),
        "delta_short": round(pick["delta"], 3), "iv": round(pick["iv"], 3) if pick["iv"] else None,
        "rendimento_annualizzato": round(ann_ret, 3),
        "protezione_ribasso_%": round(pick["mid"] / spot, 3), "open_interest": pick["oi"],
    })


def scan_cash_secured_put(symbol, spot, expiry, dte, puts, cfg, results):
    puts_ok = [q for q in puts if q["strike"] < spot]
    pick = closest_by_delta(puts_ok, cfg["csp_delta_min"], cfg["csp_delta_max"])
    if not pick:
        return
    capital = pick["strike"] * 100
    ann_ret = annualized_return(pick["mid"] * 100, capital, dte)
    if ann_ret < cfg["csp_min_annualized_return"]:
        return
    results.append({
        "strategia": "Cash-Secured Put", "simbolo": symbol, "scadenza": expiry, "dte": dte,
        "spot": round(spot, 2), "strike_short": pick["strike"], "premio": round(pick["mid"], 2),
        "delta_short": round(pick["delta"], 3), "iv": round(pick["iv"], 3) if pick["iv"] else None,
        "rendimento_annualizzato": round(ann_ret, 3),
        "breakeven": round(pick["strike"] - pick["mid"], 2), "open_interest": pick["oi"],
    })


def scan_iron_condor(symbol, spot, expiry, dte, calls, puts, cfg, results):
    width = spot * cfg["ic_wing_width_pct"]
    calls_ok = [q for q in calls if q["strike"] > spot]
    puts_ok = [q for q in puts if q["strike"] < spot]

    short_call = closest_by_delta(calls_ok, cfg["ic_short_delta_min"], cfg["ic_short_delta_max"])
    short_put = closest_by_delta(puts_ok, cfg["ic_short_delta_min"], cfg["ic_short_delta_max"])
    if not short_call or not short_put:
        return

    long_call = min((q for q in calls_ok if q["strike"] >= short_call["strike"] + width),
                     key=lambda q: q["strike"], default=None)
    long_put = max((q for q in puts_ok if q["strike"] <= short_put["strike"] - width),
                    key=lambda q: q["strike"], default=None)
    if not long_call or not long_put:
        return

    credit = (short_call["mid"] - long_call["mid"]) + (short_put["mid"] - long_put["mid"])
    max_width = max(long_call["strike"] - short_call["strike"], short_put["strike"] - long_put["strike"])
    max_loss = max_width - credit
    if credit <= 0 or max_loss <= 0:
        return
    credit_to_width = credit / max_width
    if credit_to_width < cfg["ic_min_credit_to_width"]:
        return
    pop_approx = 1 - (short_call["delta"] + short_put["delta"])

    results.append({
        "strategia": "Iron Condor", "simbolo": symbol, "scadenza": expiry, "dte": dte, "spot": round(spot, 2),
        "put_long": long_put["strike"], "put_short": short_put["strike"],
        "call_short": short_call["strike"], "call_long": long_call["strike"],
        "credito_netto": round(credit, 2), "rischio_max": round(max_loss, 2),
        "credito_su_rischio": round(credit_to_width, 3), "pop_approx": round(pop_approx, 3),
    })


def scan_credit_spread(symbol, spot, expiry, dte, calls, puts, cfg, results):
    width = spot * cfg["cs_width_pct"]

    puts_ok = [q for q in puts if q["strike"] < spot]
    short_put = closest_by_delta(puts_ok, cfg["cs_short_delta_min"], cfg["cs_short_delta_max"])
    if short_put:
        long_put = max((q for q in puts_ok if q["strike"] <= short_put["strike"] - width),
                        key=lambda q: q["strike"], default=None)
        if long_put:
            credit = short_put["mid"] - long_put["mid"]
            max_width = short_put["strike"] - long_put["strike"]
            max_loss = max_width - credit
            if credit > 0 and max_loss > 0 and credit / max_width >= cfg["cs_min_credit_to_width"]:
                results.append({
                    "strategia": "Credit Spread", "tipo": "Put (rialzista)", "simbolo": symbol,
                    "scadenza": expiry, "dte": dte, "spot": round(spot, 2),
                    "strike_long": long_put["strike"], "strike_short": short_put["strike"],
                    "credito_netto": round(credit, 2), "rischio_max": round(max_loss, 2),
                    "credito_su_rischio": round(credit / max_width, 3), "delta_short": round(short_put["delta"], 3),
                })

    calls_ok = [q for q in calls if q["strike"] > spot]
    short_call = closest_by_delta(calls_ok, cfg["cs_short_delta_min"], cfg["cs_short_delta_max"])
    if short_call:
        long_call = min((q for q in calls_ok if q["strike"] >= short_call["strike"] + width),
                         key=lambda q: q["strike"], default=None)
        if long_call:
            credit = short_call["mid"] - long_call["mid"]
            max_width = long_call["strike"] - short_call["strike"]
            max_loss = max_width - credit
            if credit > 0 and max_loss > 0 and credit / max_width >= cfg["cs_min_credit_to_width"]:
                results.append({
                    "strategia": "Credit Spread", "tipo": "Call (ribassista)", "simbolo": symbol,
                    "scadenza": expiry, "dte": dte, "spot": round(spot, 2),
                    "strike_long": long_call["strike"], "strike_short": short_call["strike"],
                    "credito_netto": round(credit, 2), "rischio_max": round(max_loss, 2),
                    "credito_su_rischio": round(credit / max_width, 3), "delta_short": round(short_call["delta"], 3),
                })


def scan_ratio_spread(symbol, spot, expiry, dte, calls, puts, cfg, results):
    ratio = cfg["rs_ratio"]

    calls_ok = [q for q in calls if q["strike"] > spot]
    long_call = closest_by_delta(calls_ok, cfg["rs_long_delta_min"], cfg["rs_long_delta_max"])
    if long_call:
        short_candidates = [q for q in calls_ok if q["strike"] > long_call["strike"]]
        short_call = closest_by_delta(short_candidates, cfg["rs_short_delta_min"], cfg["rs_short_delta_max"])
        if short_call:
            net_cost = long_call["mid"] - ratio * short_call["mid"]
            max_profit = (short_call["strike"] - long_call["strike"]) - net_cost
            results.append({
                "strategia": "Ratio Spread", "tipo": f"Call 1x{ratio} (rialzista)", "simbolo": symbol,
                "scadenza": expiry, "dte": dte, "spot": round(spot, 2),
                "strike_long": long_call["strike"], "strike_short": short_call["strike"],
                "costo_netto": round(net_cost, 2), "profitto_max_a_short": round(max_profit, 2),
                "nota": "rischio potenzialmente illimitato sopra il breakeven superiore",
            })

    puts_ok = [q for q in puts if q["strike"] < spot]
    long_put = closest_by_delta(puts_ok, cfg["rs_long_delta_min"], cfg["rs_long_delta_max"])
    if long_put:
        short_candidates = [q for q in puts_ok if q["strike"] < long_put["strike"]]
        short_put = closest_by_delta(short_candidates, cfg["rs_short_delta_min"], cfg["rs_short_delta_max"])
        if short_put:
            net_cost = long_put["mid"] - ratio * short_put["mid"]
            max_profit = (long_put["strike"] - short_put["strike"]) - net_cost
            results.append({
                "strategia": "Ratio Spread", "tipo": f"Put 1x{ratio} (ribassista)", "simbolo": symbol,
                "scadenza": expiry, "dte": dte, "spot": round(spot, 2),
                "strike_long": long_put["strike"], "strike_short": short_put["strike"],
                "costo_netto": round(net_cost, 2), "profitto_max_a_short": round(max_profit, 2),
                "nota": "rischio ampio (non oltre zero) sotto il breakeven inferiore",
            })


def scan_box_spread(symbol, spot, expiry, dte, calls, puts, cfg, results):
    width = spot * cfg["box_strike_width_pct"]
    k_low_target = spot - width / 2
    k_high_target = spot + width / 2

    call_low = closest_by_strike(calls, k_low_target)
    call_high = closest_by_strike(calls, k_high_target)
    put_low = closest_by_strike(puts, k_low_target)
    put_high = closest_by_strike(puts, k_high_target)
    if not (call_low and call_high and put_low and put_high):
        return
    if call_low["strike"] == call_high["strike"]:
        return
    if call_low["strike"] != put_low["strike"] or call_high["strike"] != put_high["strike"]:
        return

    k1, k2 = call_low["strike"], call_high["strike"]
    cost = (call_low["mid"] - call_high["mid"]) + (put_high["mid"] - put_low["mid"])
    payoff = k2 - k1
    if cost <= 0 or payoff <= 0:
        return
    T = dte / 365
    try:
        implied_rate = math.log(payoff / cost) / T
    except ValueError:
        return
    diff = implied_rate - cfg["risk_free_rate"]
    if abs(diff) < cfg["box_min_rate_diff"]:
        return
    results.append({
        "strategia": "Box Spread", "simbolo": symbol, "scadenza": expiry, "dte": dte, "spot": round(spot, 2),
        "strike_basso": k1, "strike_alto": k2, "costo_netto": round(cost, 2), "payoff_fisso": payoff,
        "tasso_implicito": round(implied_rate, 4), "tasso_riferimento": cfg["risk_free_rate"],
        "differenza_tasso": round(diff, 4),
    })


def scan_synthetic(symbol, spot, expiry, dte, calls, puts, cfg, results):
    call_atm = closest_by_strike(calls, spot)
    put_atm = closest_by_strike(puts, spot)
    if not call_atm or not put_atm or call_atm["strike"] != put_atm["strike"]:
        return
    strike = call_atm["strike"]
    T = dte / 365
    r = cfg["risk_free_rate"]
    implied_spot = strike * math.exp(-r * T) + (call_atm["mid"] - put_atm["mid"])
    deviation_pct = (implied_spot - spot) / spot
    if abs(deviation_pct) < cfg["syn_min_deviation_pct"]:
        return
    direzione = "sintetico più economico dell'azione" if implied_spot < spot else "sintetico più caro dell'azione"
    results.append({
        "strategia": "Sintetico", "simbolo": symbol, "scadenza": expiry, "dte": dte,
        "spot": round(spot, 2), "strike": strike,
        "call_mid": round(call_atm["mid"], 2), "put_mid": round(put_atm["mid"], 2),
        "prezzo_sintetico": round(implied_spot, 2), "deviazione_%": round(deviation_pct, 4),
        "direzione": direzione,
    })


# =========================================================
# Universo e scan
# =========================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


def build_universe(watchlist, use_basket, basket, max_universe):
    universe = list(dict.fromkeys(watchlist))
    if use_basket:
        if basket == "SP500":
            try:
                basket_tickers = get_sp500_tickers()
            except Exception as e:
                st.warning(f"Impossibile scaricare il paniere S&P 500 ({e}); proseguo solo con la watchlist.")
                basket_tickers = []
            for t in basket_tickers:
                if t not in universe:
                    universe.append(t)
    if len(universe) > max_universe:
        st.info(f"Universo di {len(universe)} titoli troncato a {max_universe} (alza il limite in barra laterale se vuoi coprirli tutti).")
        universe = universe[:max_universe]
    return universe


def run_scan(cfg):
    universe = build_universe(cfg["watchlist"], cfg["use_basket"], cfg["basket"], cfg["max_universe"])
    all_results = {k: [] for k in [
        "Covered Call", "Cash-Secured Put", "Iron Condor", "Credit Spread",
        "Ratio Spread", "Box Spread", "Sintetico",
    ]}
    flags = cfg["strategie_attive"]
    n = len(universe)
    progress = st.progress(0.0)
    status = st.empty()

    for idx, symbol in enumerate(universe):
        status.text(f"Scansione {symbol} ({idx + 1}/{n})...")
        try:
            yft = yf.Ticker(symbol)
            spot = get_spot_price(yft)
            if not spot or spot <= 0:
                continue
            expirations = yft.options
            if not expirations:
                continue
            expiries = pick_expiries(expirations, cfg["dte_min"], cfg["dte_max"])
            for expiry, dte in expiries:
                chain = yft.option_chain(expiry)
                calls = build_quotes(chain.calls, spot, dte, "C", cfg, cfg["strike_window_pct"])
                puts = build_quotes(chain.puts, spot, dte, "P", cfg, cfg["strike_window_pct"])
                if not calls and not puts:
                    continue
                if flags["covered_call"]:
                    scan_covered_call(symbol, spot, expiry, dte, calls, cfg, all_results["Covered Call"])
                if flags["cash_secured_put"]:
                    scan_cash_secured_put(symbol, spot, expiry, dte, puts, cfg, all_results["Cash-Secured Put"])
                if flags["iron_condor"]:
                    scan_iron_condor(symbol, spot, expiry, dte, calls, puts, cfg, all_results["Iron Condor"])
                if flags["credit_spread"]:
                    scan_credit_spread(symbol, spot, expiry, dte, calls, puts, cfg, all_results["Credit Spread"])
                if flags["ratio_spread"]:
                    scan_ratio_spread(symbol, spot, expiry, dte, calls, puts, cfg, all_results["Ratio Spread"])
                if flags["box_spread"]:
                    scan_box_spread(symbol, spot, expiry, dte, calls, puts, cfg, all_results["Box Spread"])
                if flags["synthetic"]:
                    scan_synthetic(symbol, spot, expiry, dte, calls, puts, cfg, all_results["Sintetico"])
            time.sleep(cfg["pacing_sleep_sec"])
        except Exception as e:
            st.warning(f"[{symbol}] errore: {e}")
        progress.progress((idx + 1) / max(n, 1))

    progress.empty()
    status.empty()
    return all_results


# =========================================================
# Costruttore di payoff
# =========================================================

def leg_payoff(spot_array, leg):
    if leg["tipo"] == "stock":
        return leg["qty"] * (spot_array - leg["premio"])
    elif leg["tipo"] == "call":
        intrinsic = np.maximum(spot_array - leg["strike"], 0)
        return leg["qty"] * 100 * (intrinsic - leg["premio"])
    elif leg["tipo"] == "put":
        intrinsic = np.maximum(leg["strike"] - spot_array, 0)
        return leg["qty"] * 100 * (intrinsic - leg["premio"])
    else:
        raise ValueError(f"Tipo di leg non valido: {leg['tipo']}")


def total_payoff(spot_array, legs):
    total = np.zeros_like(spot_array, dtype=float)
    for leg in legs:
        total = total + leg_payoff(spot_array, leg)
    return total


def plot_payoff_fig(legs, center=None, range_pct=0.30, n_points=400):
    strikes = [l["strike"] for l in legs if l.get("strike") is not None]
    stock_legs = [l["premio"] for l in legs if l["tipo"] == "stock"]
    if center is None:
        center = stock_legs[0] if stock_legs else (sum(strikes) / len(strikes) if strikes else 100)

    spot_range = np.linspace(center * (1 - range_pct), center * (1 + range_pct), n_points)
    pl = total_payoff(spot_range, legs)

    breakevens = []
    for i in range(len(pl) - 1):
        if pl[i] == 0:
            breakevens.append(spot_range[i])
        elif pl[i] < 0 < pl[i + 1] or pl[i] > 0 > pl[i + 1]:
            x0, x1, y0, y1 = spot_range[i], spot_range[i + 1], pl[i], pl[i + 1]
            breakevens.append(x0 - y0 * (x1 - x0) / (y1 - y0))

    max_profit, min_profit = pl.max(), pl.min()
    profit_open = pl[-1] > pl[-2]
    loss_open = pl[0] < pl[1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spot_range, pl, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    for be in breakevens:
        ax.axvline(be, color="gray", linestyle="--", linewidth=0.8)
        ax.annotate(f"BE {be:.2f}", (be, 0), textcoords="offset points", xytext=(4, 8), fontsize=8)
    ax.set_title("Payoff a scadenza")
    ax.set_xlabel("Prezzo del sottostante")
    ax.set_ylabel("Profitto / Perdita ($)")
    ax.grid(alpha=0.3)

    summary = {
        "max_profit": max_profit, "min_profit": min_profit,
        "profit_open": profit_open, "loss_open": loss_open,
        "breakevens": breakevens,
    }
    return fig, summary


# =========================================================
# UI — Barra laterale
# =========================================================

st.sidebar.header("⚙️ Configurazione")

watchlist_input = st.sidebar.text_area("Watchlist (ticker separati da virgola)", "AAPL, MSFT, NVDA, AMD, TSLA")
watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

use_basket = st.sidebar.checkbox("Includi un paniere ampio", value=True)
basket = st.sidebar.selectbox("Paniere", ["SP500"], disabled=not use_basket)
max_universe = st.sidebar.slider("Numero massimo di titoli da scansionare", 5, 200, 40, step=5)

st.sidebar.subheader("Scadenze e strike")
dte_min, dte_max = st.sidebar.slider("Giorni a scadenza (DTE)", 1, 120, (20, 45))
strike_window_pct = st.sidebar.slider("Finestra strike attorno allo spot (%)", 5, 50, 20) / 100

st.sidebar.subheader("Parametri di calcolo")
risk_free_rate = st.sidebar.number_input("Tasso privo di rischio annuo", 0.0, 0.20, 0.045, step=0.005, format="%.3f")
dividend_yield = st.sidebar.number_input("Dividend yield (semplificato, uguale per tutti)", 0.0, 0.20, 0.0, step=0.005, format="%.3f")
min_open_interest = st.sidebar.number_input("Open interest minimo", 0, 10000, 10, step=5)
max_bid_ask_spread_pct = st.sidebar.slider("Spread bid/ask massimo (%)", 1, 100, 25) / 100

st.sidebar.subheader("Strategie attive")
flags = {
    "covered_call": st.sidebar.checkbox("Covered Call", True),
    "cash_secured_put": st.sidebar.checkbox("Cash-Secured Put", True),
    "iron_condor": st.sidebar.checkbox("Iron Condor", True),
    "credit_spread": st.sidebar.checkbox("Credit Spread", True),
    "ratio_spread": st.sidebar.checkbox("Ratio Spread", True),
    "box_spread": st.sidebar.checkbox("Box Spread", True),
    "synthetic": st.sidebar.checkbox("Sintetico", True),
}

with st.sidebar.expander("Parametri avanzati per strategia"):
    cc_delta_min, cc_delta_max = st.slider("Covered Call — range delta", 0.05, 0.60, (0.15, 0.35))
    cc_min_annualized_return = st.number_input("Covered Call — rendimento annualizzato minimo", 0.0, 2.0, 0.10, step=0.01)

    csp_delta_min, csp_delta_max = st.slider("Cash-Secured Put — range delta", 0.05, 0.60, (0.15, 0.35))
    csp_min_annualized_return = st.number_input("Cash-Secured Put — rendimento annualizzato minimo", 0.0, 2.0, 0.10, step=0.01)

    ic_short_delta_min, ic_short_delta_max = st.slider("Iron Condor — range delta gambe corte", 0.02, 0.40, (0.10, 0.20))
    ic_wing_width_pct = st.slider("Iron Condor — ampiezza ala (%)", 1, 30, 5) / 100
    ic_min_credit_to_width = st.slider("Iron Condor — credito minimo su rischio", 0.0, 1.0, 0.20)

    cs_short_delta_min, cs_short_delta_max = st.slider("Credit Spread — range delta gamba corta", 0.05, 0.50, (0.15, 0.30))
    cs_width_pct = st.slider("Credit Spread — ampiezza (%)", 1, 30, 5) / 100
    cs_min_credit_to_width = st.slider("Credit Spread — credito minimo su rischio", 0.0, 1.0, 0.20)

    rs_ratio = st.number_input("Ratio Spread — rapporto (1xN)", 2, 5, 2)
    rs_long_delta_min, rs_long_delta_max = st.slider("Ratio Spread — delta gamba lunga", 0.10, 0.70, (0.40, 0.55))
    rs_short_delta_min, rs_short_delta_max = st.slider("Ratio Spread — delta gambe corte", 0.02, 0.50, (0.15, 0.25))

    box_strike_width_pct = st.slider("Box Spread — ampiezza tra le strike (%)", 2, 30, 10) / 100
    box_min_rate_diff = st.number_input("Box Spread — differenza tasso minima da segnalare", 0.0, 0.20, 0.01, step=0.005, format="%.3f")

    syn_min_deviation_pct = st.number_input("Sintetico — deviazione minima da segnalare (%)", 0.0, 0.20, 0.01, step=0.005, format="%.3f")

top_n = st.sidebar.slider("Righe massime per tabella", 5, 50, 15)
pacing_sleep_sec = st.sidebar.slider("Pausa tra un titolo e l'altro (secondi)", 0.0, 3.0, 0.5, step=0.1)

cfg = {
    "watchlist": watchlist, "use_basket": use_basket, "basket": basket, "max_universe": max_universe,
    "dte_min": dte_min, "dte_max": dte_max, "strike_window_pct": strike_window_pct,
    "risk_free_rate": risk_free_rate, "dividend_yield": dividend_yield,
    "min_open_interest": min_open_interest, "max_bid_ask_spread_pct": max_bid_ask_spread_pct,
    "strategie_attive": flags,
    "cc_delta_min": cc_delta_min, "cc_delta_max": cc_delta_max, "cc_min_annualized_return": cc_min_annualized_return,
    "csp_delta_min": csp_delta_min, "csp_delta_max": csp_delta_max, "csp_min_annualized_return": csp_min_annualized_return,
    "ic_short_delta_min": ic_short_delta_min, "ic_short_delta_max": ic_short_delta_max,
    "ic_wing_width_pct": ic_wing_width_pct, "ic_min_credit_to_width": ic_min_credit_to_width,
    "cs_short_delta_min": cs_short_delta_min, "cs_short_delta_max": cs_short_delta_max,
    "cs_width_pct": cs_width_pct, "cs_min_credit_to_width": cs_min_credit_to_width,
    "rs_ratio": rs_ratio, "rs_long_delta_min": rs_long_delta_min, "rs_long_delta_max": rs_long_delta_max,
    "rs_short_delta_min": rs_short_delta_min, "rs_short_delta_max": rs_short_delta_max,
    "box_strike_width_pct": box_strike_width_pct, "box_min_rate_diff": box_min_rate_diff,
    "syn_min_deviation_pct": syn_min_deviation_pct,
    "pacing_sleep_sec": pacing_sleep_sec,
}

run_button = st.sidebar.button("🔍 Esegui Scan", type="primary", width='stretch')

# =========================================================
# UI — Corpo principale
# =========================================================

st.title("📊 Screener Multi-Strategia Opzioni")
st.caption(
    "Dati via Yahoo Finance (yfinance) — indicativi/ritardati. "
    "Il delta è calcolato con Black-Scholes a partire dalla implied volatility fornita da Yahoo, "
    "non è il delta ufficiale del broker."
)

if run_button:
    with st.spinner("Scansione in corso..."):
        st.session_state["results"] = run_scan(cfg)
        st.session_state["top_n"] = top_n

if "results" not in st.session_state:
    st.info("Configura i parametri nella barra laterale e premi **Esegui Scan** per iniziare.")
else:
    results = st.session_state["results"]
    tn = st.session_state.get("top_n", 15)

    tab_names = [
        "Covered Call", "Cash-Secured Put", "Iron Condor", "Credit Spread",
        "Ratio Spread", "Box Spread", "Sintetico", "🧮 Costruttore Payoff",
    ]
    tabs = st.tabs(tab_names)

    def render_table(tab, name, sort_col):
        with tab:
            rows = results.get(name, [])
            if not rows:
                st.info("Nessun risultato con i filtri attuali (o strategia disattivata in barra laterale).")
                return
            df = pd.DataFrame(rows)
            if sort_col == "differenza_tasso":
                order = df[sort_col].abs().sort_values(ascending=False).index
                df = df.loc[order].head(tn).reset_index(drop=True)
            else:
                df = df.sort_values(sort_col, ascending=False).head(tn).reset_index(drop=True)
            st.dataframe(df, width='stretch', hide_index=True)

    render_table(tabs[0], "Covered Call", "rendimento_annualizzato")
    render_table(tabs[1], "Cash-Secured Put", "rendimento_annualizzato")
    render_table(tabs[2], "Iron Condor", "credito_su_rischio")
    render_table(tabs[3], "Credit Spread", "credito_su_rischio")
    render_table(tabs[4], "Ratio Spread", "profitto_max_a_short")
    render_table(tabs[5], "Box Spread", "differenza_tasso")
    render_table(tabs[6], "Sintetico", "deviazione_%")

    with tabs[7]:
        st.subheader("Costruttore manuale di strategia")
        st.caption(
            "qty positivo = long (comprata), negativo = short (venduta). "
            "Per le opzioni qty è in numero di contratti (1 = 100 azioni); per l'azione qty è in numero di azioni."
        )
        default_legs = pd.DataFrame([
            {"tipo": "stock", "strike": None, "premio": 100.0, "qty": 100},
            {"tipo": "put", "strike": 95.0, "premio": 1.5, "qty": 1},
            {"tipo": "call", "strike": 105.0, "premio": 1.8, "qty": -1},
        ])
        legs_df = st.data_editor(
            default_legs, num_rows="dynamic", width='stretch',
            column_config={
                "tipo": st.column_config.SelectboxColumn(options=["stock", "call", "put"], required=True),
                "strike": st.column_config.NumberColumn(help="Lascia vuoto per la gamba azionaria"),
                "premio": st.column_config.NumberColumn(help="Prezzo per azione pagato/incassato"),
                "qty": st.column_config.NumberColumn(help="Contratti (opzioni) o azioni (stock); negativo = short"),
            },
        )
        if st.button("Disegna payoff"):
            legs = legs_df.dropna(subset=["premio", "qty", "tipo"]).to_dict("records")
            legs = [{**l, "strike": None if l["tipo"] == "stock" else l["strike"]} for l in legs]
            if not legs:
                st.warning("Aggiungi almeno una gamba valida.")
            else:
                fig, summary = plot_payoff_fig(legs)
                st.pyplot(fig)
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Profitto max (nell'intervallo)", f"${summary['max_profit']:.2f}",
                    "aperto oltre il grafico" if summary["profit_open"] else None,
                )
                col2.metric(
                    "Perdita max (nell'intervallo)", f"${summary['min_profit']:.2f}",
                    "aperta oltre il grafico" if summary["loss_open"] else None,
                )
                col3.metric("Breakeven", ", ".join(f"{b:.2f}" for b in summary["breakevens"]) or "nessuno nell'intervallo")
