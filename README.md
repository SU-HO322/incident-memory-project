# 🧠 Incident Memory Agent

> AI that remembers how your team fixed it last time.

---

## What It Does

Paste a production error log or stack trace → the system searches your team's incident history using semantic similarity → Gemini synthesizes a recommendation grounded in real, proven fixes.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI | Gemini 2.5 Flash + text-embedding-004 |
| Database | Supabase PostgreSQL + pgvector |
| Backend | Python 3.12+ |

---

## 🗄️ Supabase Setup (Complete Guide)

### Step 1 — Create a Supabase Account & Project

1. Go to [supabase.com](https://supabase.com) and sign up (free)
2. Click **"New Project"**
3. Fill in:
   - **Name**: `incident-memory-agent`
   - **Database Password**: choose a strong password and **save it** — you'll need it later
   - **Region**: pick the one closest to you
4. Click **"Create new project"** — wait ~2 minutes for provisioning

### Step 2 — Enable pgvector Extension

1. In your Supabase project, go to **Database → Extensions** (left sidebar)
2. Search for `vector`
3. Toggle it **ON**

   > The `schema.sql` file also runs `CREATE EXTENSION IF NOT EXISTS vector` as a safety net, but enabling it in the dashboard first is best practice.

### Step 3 — Run the Schema

1. Go to **SQL Editor** (left sidebar, looks like `</>`)
2. Click **"New query"**
3. Open the file [`db/schema.sql`](db/schema.sql) from this repo
4. Copy the **entire contents** and paste it into the SQL Editor
5. Click **"Run"** (or press `Ctrl+Enter`)
6. You should see: `Success. No rows returned`

**Verify it worked** — run this in a new SQL Editor query:
```sql
SELECT table_name FROM information_schema.tables WHERE table_name = 'incidents';
SELECT routine_name FROM information_schema.routines WHERE routine_name = 'search_similar_incidents';
```
Both queries should return one row each.

### Step 4 — Get Your API Keys

#### Supabase URL + Anon Key
1. Go to **Project Settings → API** (gear icon → API)
2. Copy:
   - **Project URL** → this is your `SUPABASE_URL`
   - **Project API Keys → `anon` `public`** → this is your `SUPABASE_KEY`

#### Database Connection String (for pgvector direct access)
1. Go to **Project Settings → Database**
2. Scroll to **"Connection string"**
3. Select **"URI"** tab
4. Copy the string — it looks like:
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.abcdefgh.supabase.co:5432/postgres
   ```
5. Replace `[YOUR-PASSWORD]` with the password you set in Step 1

### Step 5 — Get Your Gemini API Key

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Copy the key

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/incident-memory-agent.git
cd incident-memory-agent
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Open `.env` and fill in your real keys (see Supabase Setup above).

### 5. Verify Setup (Smoke Test)
Run the diagnostic script to verify database connection and Gemini API:
```bash
python scripts/smoke_test.py
```

### 6. Seed Sample Incidents for Demo
Populate the database with 5 realistic production incidents:
```bash
# Preview incidents without modifying database:
python scripts/seed_data.py --dry-run

# Insert incidents with 768-dim embeddings:
python scripts/seed_data.py
```

### 7. Run the App
```bash
streamlit run app/main.py
```

---

## Project Structure

```
incident-memory-agent/
├── app/                       # Streamlit UI
│   ├── main.py                # App entry point & landing page
│   ├── components/            # Reusable UI widgets
│   │   ├── confidence_badge.py
│   │   └── incident_card.py
│   └── pages/                 # Multi-page views
│       ├── 01_search.py       # Semantic search & AI Advisor
│       ├── 02_add_incident.py # Incident logger form
│       └── 03_dashboard.py    # Analytics & incident explorer
├── config/                    # Settings & env loader (psycopg3 + Gemini)
├── db/                        # PostgreSQL schema with pgvector HNSW index
├── models/                    # Pydantic data models & validation
├── scripts/                   # CLI tools
│   ├── seed_data.py           # Populates 5 realistic production incidents
│   └── smoke_test.py          # End-to-end environment & pipeline diagnostics
├── services/                  # Core pipeline services
│   ├── ai_advisor.py          # RAG recommendations with Gemini 2.5 Flash
│   ├── embedder.py            # text-embedding-004 vectorizer
│   ├── incident_service.py    # Main orchestration pipeline
│   └── vector_store.py        # Direct psycopg3 + pgvector interface
└── tests/                     # 48 unit tests (100% passing)
```

---

## Running Tests

```bash
# Run the complete test suite (runs offline with mock fixtures)
pytest tests/ -v
```

---

## ⚠️ Important Notes

- **Never commit `.env`** — it contains your real API keys
- The system only recommends based on **real stored incidents** — it never invents fixes
- Seed the database with real incidents before demoing (see `02_add_incident` page)
