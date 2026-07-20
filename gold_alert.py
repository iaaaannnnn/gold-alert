"""
Gold Long-Term Buy Alert Bot
------------------------------
Runs once daily (via GitHub Actions cron).

Logic:
1. Fetch current gold spot price + recent historical prices from GoldAPI.
2. Calculate the 50-day moving average and % deviation from it.
3. If price is >= DIP_THRESHOLD (default 5%) below the 50-day MA:
     a. Fetch recent gold/macro news headlines from NewsAPI.
     b. Send headlines to NVIDIA NIM (Llama 3.3 70B) for a bullish/
        bearish/neutral sentiment read with reasoning.
     c. If sentiment is neutral-to-bullish, send an email alert via Resend.
4. Otherwise, exit quietly (no email).

Required environment variables (set as GitHub Actions secrets):
    GOLDAPI_KEY
    NEWSAPI_KEY
    RESEND_API_KEY
    NIM_API_KEY
    ALERT_EMAIL          - destination inbox
    RESEND_FROM_EMAIL    - verified "from" address in Resend (or their test domain)

Optional environment variables:
    DIP_THRESHOLD_PCT    - default "5" (percent below MA to trigger a check)
"""

import os
import sys
import json
import statistics
from datetime import datetime, timedelta, timezone

import requests

# ---------- Config ----------
GOLDAPI_KEY = os.environ["GOLDAPI_KEY"]
NEWSAPI_KEY = os.environ["NEWSAPI_KEY"]
RESEND_API_KEY = os.environ["RESEND_API_KEY"]
NIM_API_KEY = os.environ["NIM_API_KEY"]
ALERT_EMAIL = os.environ["ALERT_EMAIL"]
RESEND_FROM_EMAIL = os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev")

DIP_THRESHOLD_PCT = float(os.environ.get("DIP_THRESHOLD_PCT", "5"))
MA_WINDOW_DAYS = 50

GOLDAPI_BASE = "https://www.goldapi.io/api"
NEWSAPI_BASE = "https://newsapi.org/v2/everything"
NIM_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
RESEND_BASE = "https://api.resend.com/emails"


# ---------- Step 1: Gold price data ----------
def get_current_gold_price():
    """Fetch current XAU/USD spot price from GoldAPI."""
    headers = {"x-access-token": GOLDAPI_KEY, "Content-Type": "application/json"}
    resp = requests.get(f"{GOLDAPI_BASE}/XAU/USD", headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data["price"]


def get_historical_gold_prices(days=MA_WINDOW_DAYS):
    """
    Fetch historical daily prices from GoldAPI (one call per day is the
    free-tier-friendly approach). GoldAPI's historical endpoint format:
    GET /api/XAU/USD/YYYYMMDD
    """
    prices = []
    today = datetime.now(timezone.utc).date()
    headers = {"x-access-token": GOLDAPI_KEY, "Content-Type": "application/json"}

    for i in range(1, days + 1):
        day = today - timedelta(days=i)
        date_str = day.strftime("%Y%m%d")
        try:
            resp = requests.get(
                f"{GOLDAPI_BASE}/XAU/USD/{date_str}", headers=headers, timeout=20
            )
            if resp.status_code == 200:
                data = resp.json()
                if "price" in data:
                    prices.append(data["price"])
        except requests.RequestException:
            continue

    return prices


# ---------- Step 2: News ----------
def get_gold_news(page_size=10):
    """Fetch recent gold/macro-relevant headlines from NewsAPI."""
    params = {
        "q": (
            '"gold price" OR "gold market" OR "Federal Reserve" OR '
            '"interest rate" OR inflation OR "US dollar"'
        ),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }
    resp = requests.get(NEWSAPI_BASE, params=params, timeout=20)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    headlines = [
        f"- {a['title']} ({a.get('source', {}).get('name', 'unknown source')})"
        for a in articles
        if a.get("title")
    ]
    return headlines


# ---------- Step 3: Sentiment via NVIDIA NIM ----------
def get_sentiment_read(headlines, price, ma, deviation_pct):
    """Ask NVIDIA NIM (Llama 3.3 70B) for a bullish/bearish/neutral read."""
    headlines_text = "\n".join(headlines) if headlines else "No recent headlines found."

    prompt = f"""You are a cautious long-term gold market analyst.

Current gold spot price: ${price:.2f}
50-day moving average: ${ma:.2f}
Deviation from moving average: {deviation_pct:.2f}% (negative = below average)

Recent macro/gold-related headlines:
{headlines_text}

Task: Based ONLY on the above, give a read on whether this price dip looks
like a reasonable long-term buying opportunity, or a sign of a deeper
structural decline to avoid.

Respond ONLY in this exact JSON format, nothing else, no markdown fences:
{{
  "sentiment": "bullish" | "neutral" | "bearish",
  "reasoning": "2-3 sentence explanation in plain language"
}}
"""

    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 400,
    }

    resp = requests.post(NIM_BASE, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip accidental markdown fences if the model adds them anyway
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"sentiment": "neutral", "reasoning": content}

    return parsed


# ---------- Step 4: Email via Resend ----------
def send_status_email(
    price,
    ma,
    deviation_pct,
    is_buy_signal,
    sentiment=None,
    reasoning=None,
    headlines=None,
):
    """
    Sends a daily status email either way.
    - is_buy_signal=True  -> labeled as a potential buy signal
    - is_buy_signal=False -> labeled as no potential buy, still shows the numbers
    """
    headlines = headlines or []

    if is_buy_signal:
        subject = f"🟢 Potential Buy Signal: ${price:.2f} ({deviation_pct:.1f}% below 50-day MA)"
        heading = "Potential Long-Term Gold Buy Signal"
    else:
        subject = f"⚪ No Potential Buy: ${price:.2f} ({deviation_pct:.1f}% vs 50-day MA)"
        heading = "No Potential Buy Today"

    sentiment_block = ""
    headlines_block = ""
    if sentiment is not None:
        sentiment_block = f"""
    <p><b>AI sentiment read:</b> {sentiment.upper()}</p>
    <p><b>Reasoning:</b> {reasoning}</p>
    """
    if headlines:
        headlines_html = "".join(f"<li>{h[2:]}</li>" for h in headlines[:8])
        headlines_block = f"""
    <h3>Headlines considered:</h3>
    <ul>{headlines_html}</ul>
    """

    html_body = f"""
    <h2>{heading}</h2>
    <p><b>Current price:</b> ${price:.2f}</p>
    <p><b>50-day moving average:</b> ${ma:.2f}</p>
    <p><b>Deviation:</b> {deviation_pct:.2f}% {'below' if deviation_pct < 0 else 'above'} average</p>
    <p><b>Buy threshold:</b> -{DIP_THRESHOLD_PCT}% below 50-day MA</p>
    {sentiment_block}
    {headlines_block}
    <p style="color:gray;font-size:12px;">
      This is an automated daily check based on price and news heuristics only.
      Not financial advice — do your own research before trading.
    </p>
    """

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "from": RESEND_FROM_EMAIL,
        "to": [ALERT_EMAIL],
        "subject": subject,
        "html": html_body,
    }

    resp = requests.post(RESEND_BASE, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    print(f"Email sent successfully ({'BUY SIGNAL' if is_buy_signal else 'no signal'}).")


# ---------- Main ----------
def main():
    print("Fetching current gold price...")
    price = get_current_gold_price()

    print("Fetching historical prices for moving average...")
    history = get_historical_gold_prices()

    if len(history) < 10:
        print(
            f"Not enough historical data ({len(history)} points) to compute a "
            "reliable moving average. Exiting without alert."
        )
        sys.exit(0)

    ma = statistics.mean(history)
    deviation_pct = ((price - ma) / ma) * 100

    print(f"Price: ${price:.2f} | 50-day MA: ${ma:.2f} | Deviation: {deviation_pct:.2f}%")

    if deviation_pct > -DIP_THRESHOLD_PCT:
        print(
            f"Deviation ({deviation_pct:.2f}%) does not meet the "
            f"-{DIP_THRESHOLD_PCT}% threshold. Sending 'no potential buy' email."
        )
        send_status_email(price, ma, deviation_pct, is_buy_signal=False)
        sys.exit(0)

    print("Threshold met. Fetching news...")
    headlines = get_gold_news()

    print("Requesting sentiment read from NVIDIA NIM...")
    result = get_sentiment_read(headlines, price, ma, deviation_pct)
    sentiment = result.get("sentiment", "neutral").lower()
    reasoning = result.get("reasoning", "No reasoning provided.")

    print(f"Sentiment: {sentiment} | Reasoning: {reasoning}")

    if sentiment in ("bullish", "neutral"):
        send_status_email(
            price, ma, deviation_pct, is_buy_signal=True,
            sentiment=sentiment, reasoning=reasoning, headlines=headlines,
        )
    else:
        print("Sentiment is bearish. Sending 'no potential buy' email (dip may reflect further decline).")
        send_status_email(
            price, ma, deviation_pct, is_buy_signal=False,
            sentiment=sentiment, reasoning=reasoning, headlines=headlines,
        )


if __name__ == "__main__":
    main()
