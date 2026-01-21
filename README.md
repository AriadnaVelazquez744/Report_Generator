Report Generator - Simplified Stack
===================================

What you get
---

- FastAPI backend that reads/writes JSON files (no database needed).
- Frontend in TypeScript/HTML/CSS (already built to `app.js`, `auth.js`, `login.js`).
- Data lives in `Data/Data_users/users.json`, `Data/current_session.json`, and `Data/Data_articles/**.json`.

Prereqs
---

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) or pip
- Node (only if you want to recompile the TypeScript)

Install deps
---

```bash
# optional: download NLP resources the project uses
uv run python scripts/setup_nltk.py
uv run python -m spacy download es_core_news_sm
```

Configure Telegram credentials (optional)
---

To enable automatic news updates from Telegram, create a `.env` file in the project root with:

```bash
api_id=YOUR_TELEGRAM_API_ID
api_hash=YOUR_TELEGRAM_API_HASH
tg_group_username=teleSUR_tv  # optional, defaults to "teleSUR_tv"
```

**How to get Telegram API credentials:**

1. Go to <https://my.telegram.org/apps>
2. Log in with your phone number
3. Create a new application (or use existing)
4. Copy the `api_id` and `api_hash` to your `.env` file

**Note:** You don't need a bot token - these are user API credentials. The system will use your Telegram account to read messages from public groups/channels.

If credentials are not configured, the automatic update will be skipped silently.

Run the API
---

```bash
uv run python scripts/run_api.py
# API lives at http://localhost:8000
```

*Wait for the API to finish loading before opening the frontend.*

Serve the frontend
---

```bash
cd src/frontend
# if you edit .ts files, rebuild with: tsc
python3 -m http.server 5500  # or any static server
# open http://localhost:5500/login.html
```

How data flows
---

- Register/login hits `/api/users/register` and `/api/users/login`.
- Session messages use `/api/session`, `/api/session/message`, `/api/session/clear` and are stored in `Data/current_session.json`.
- Report requests go to `/recommendations/generate`; PDF downloads hit `/reports/generate-pdf` and are saved to `reportes_pdf/`.

Notes
---

- If you change `Data/Data_articles/`, restart the API so it rebuilds its cache.
- Passwords are stored in plain text in `Data/Data_users/users.json` for simplicity; do not use real credentials.
