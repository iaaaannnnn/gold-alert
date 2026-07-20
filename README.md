# Gold Long-Term Buy Alert Bot

Checks gold price once daily (6:00 AM Philippine time) against its 50-day
moving average. If the price is 5%+ below the average, it pulls recent
gold/macro news, gets a sentiment read from NVIDIA NIM (Llama 3.3 70B), and
emails you (via Resend) only if the dip looks like a reasonable long-term
entry rather than a sign of further decline.

## Setup

1. Create these repo secrets under **Settings → Secrets and variables →
   Actions**:
   - `GOLDAPI_KEY` — from https://www.goldapi.io
   - `NEWSAPI_KEY` — from https://newsapi.org
   - `RESEND_API_KEY` — from https://resend.com
   - `NIM_API_KEY` — from https://build.nvidia.com
   - `ALERT_EMAIL` — the inbox you want alerts sent to
   - `RESEND_FROM_EMAIL` — a verified sending address in Resend
     (or use `onboarding@resend.dev` for testing, no verification needed)

2. Push this repo to GitHub (see steps below).

3. Go to the **Actions** tab → select "Daily Gold Buy Signal Check" →
   click **Run workflow** to test it manually before waiting for the
   scheduled run.

4. It will otherwise run automatically every day at 22:00 UTC
   (6:00 AM Philippine time).

## Tuning

- Change `DIP_THRESHOLD_PCT` in `gold-check.yml` to make alerts more or
  less frequent (lower = more alerts, higher = fewer/bigger dips only).
- Edit the prompt in `get_sentiment_read()` inside `gold_alert.py` to
  adjust how the AI reasons about the news.

## Disclaimer

This is a personal heuristic tool, not financial advice. Price + news
sentiment signals are simplistic by nature — always do your own research
before acting on an alert.
