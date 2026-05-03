# Fake Product Review Detection System

An industry-level ML system that detects fake product reviews using NLP classification, sentiment analysis, and reviewer behavior modeling.

## Features

- **NLP Classification** — TF-IDF + Logistic Regression trained on real/fake review patterns
- **Sentiment vs Rating Mismatch** — Detects when a 5-star review reads negatively (or vice versa)
- **Reviewer Behavior Analysis** — Flags burst activity, new accounts, rating extremism, generic language
- **Explainability** — Per-review breakdown of NLP features and behavioral red flags
- **Dashboard** — History of analyzed reviews, aggregate stats, sentiment/flag charts
- **REST API** — FastAPI backend with full OpenAPI docs at `/api/docs`

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Backend | Python 3.11, FastAPI, scikit-learn, NLTK VADER |
| Frontend | React 19, Vite, Tailwind CSS, Recharts, Framer Motion |
| Database | PostgreSQL (optional — graceful fallback without it) |
| Deployment | Vercel (frontend + Python serverless) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ml/analyze` | Analyze a single review |
| `POST` | `/ml/batch` | Analyze multiple reviews |
| `GET` | `/ml/demo` | Pre-analyzed demo reviews |
| `GET` | `/ml/reviews` | All analyzed reviews (requires DB) |
| `GET` | `/ml/stats` | Aggregate stats (requires DB) |
| `GET` | `/ml/health` | Health check |

### Sample Request

```bash
curl -X POST https://your-app.vercel.app/ml/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "review_text": "WOW!! AMAZING product!! BEST PURCHASE EVER!! Changed my life completely!!!",
    "rating": 5,
    "reviewer_id": "user_123",
    "product": "Wireless Earbuds"
  }'
```

### Sample Response

```json
{
  "is_fake": true,
  "fake_score": 0.89,
  "confidence": "high",
  "sentiment": "positive",
  "sentiment_score": 0.92,
  "sentiment_mismatch": false,
  "behavioral_flags": ["excessive_punctuation", "excessive_caps", "generic_language", "suspiciously_enthusiastic"],
  "nlp_features": {
    "word_count": 14,
    "exclamation_count": 6,
    "caps_ratio": 0.31,
    "generic_phrase_hits": 3,
    "lexical_diversity": 0.79
  }
}
```

## Deploy to Vercel (via GitHub)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/fake-review-detector.git
git push -u origin main
```

### Step 2 — Import to Vercel

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **"Import Git Repository"** and select your repo
3. Vercel auto-detects the `vercel.json` config — no manual settings needed
4. Click **Deploy**

### Step 3 — Add Database (Optional)

For review history and stats, add a PostgreSQL database:

1. In your Vercel project → **Storage** tab → **Create Database** → **Postgres**
2. Copy the `DATABASE_URL` connection string
3. Go to **Settings → Environment Variables** → add `DATABASE_URL`
4. Redeploy

The app works without a database — the `/ml/analyze` and `/ml/demo` endpoints are fully functional without one. Only `/ml/reviews` and `/ml/stats` require it.

## Local Development

### Prerequisites

- Node.js 20+, pnpm 9+
- Python 3.11+

### Setup

```bash
# Install Node.js dependencies
pnpm install

# Install Python dependencies
pip install -r api/requirements.txt

# Start Python ML API (runs on :8090)
bash artifacts/review-api/start.sh &

# Start React frontend (runs on :20446)
pnpm --filter @workspace/review-detector run dev
```

Open [http://localhost:80](http://localhost:80)

## ML Model Details

### Features Used

**Text Features**
- TF-IDF unigrams + bigrams (3,000 features)
- Exclamation mark density
- ALL-CAPS ratio
- Generic/template phrase detection
- Lexical diversity (type-token ratio)
- Review length

**Behavioral Features**
- New account burst (many reviews, account < 7 days old)
- High review frequency (5+ reviews in 30 days)
- Rating extremism (always 5 or always 1)
- Sentiment vs star-rating mismatch

### Evaluation

The model is trained on synthetic data with deliberately distinct patterns. For production use, replace or augment with labeled data from:
- [Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html)
- [Yelp Open Dataset](https://www.yelp.com/dataset)
- [FakeReviewNet](https://github.com/mahmedbaig/FakeReviewNet)

### Extending the Model

To swap in a stronger model (XGBoost, LightGBM, BERT):

```python
# In api/index.py — replace the Pipeline:
from xgboost import XGBClassifier

nlp_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
])
```

## Project Structure

```
├── api/
│   ├── index.py          # FastAPI ML backend (Vercel serverless)
│   └── requirements.txt  # Python dependencies
├── artifacts/
│   ├── review-detector/  # React + Vite frontend
│   └── review-api/       # Local Python service (for Replit dev)
├── lib/                  # Shared TypeScript libraries
├── vercel.json           # Vercel deployment config
└── README.md
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | No | PostgreSQL connection string for review history |
| `BASE_PATH` | Build only | URL base path (set to `/` automatically by Vercel) |

## License

MIT
