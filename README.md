Report Generator - Simplified Stack
===================================

What you get
---

- FastAPI backend that reads/writes JSON files (no database needed).
- Frontend in TypeScript/HTML/CSS (already built to `app.js`, `auth.js`, `login.js`).
- Data lives in `Data_users/users.json`, `current_session.json`, and `Data_articles/**.json`.

Prereqs
---

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) or pip
- Node (only if you want to recompile the TypeScript)

Install deps
---

```bash
uv sync
# optional: download NLP resources the project uses
uv run python scripts/setup_nltk.py
uv run python -m spacy download es_core_news_sm
```

Run the API
---

```bash
uv run python scripts/run_api.py
# API lives at http://localhost:8000
```

Serve the frontend
---

```bash
cd src/frontend
# if you edit .ts files, rebuild with: tsc
python -m http.server 5500  # or any static server
# open http://localhost:5500/login.html
```

How data flows
---

- Register/login hits `/api/users/register` and `/api/users/login`.
- Session messages use `/api/session`, `/api/session/message`, `/api/session/clear` and are stored in `current_session.json`.
- Report requests go to `/recommendations/generate`; PDF downloads hit `/reports/generate-pdf` and are saved to `reportes_pdf/`.

Notes
---

- If you change `Data_articles/`, restart the API so it rebuilds its cache.
- Passwords are stored in plain text in `Data_users/users.json` for simplicity; do not use real credentials.

