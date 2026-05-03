# Fake Product Review Detection System

## Overview

Industry-level ML system for detecting fake product reviews using NLP, sentiment analysis, and reviewer behavior modeling. Built as a full-stack portfolio project.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **ML Backend**: Python 3.11, FastAPI, scikit-learn, NLTK VADER
- **Frontend**: React 19, Vite, Tailwind CSS, Recharts, Framer Motion, wouter
- **Database**: PostgreSQL + Drizzle ORM (TypeScript side)
- **API framework**: Express 5 (Node.js), FastAPI (Python ML)
- **Deployment**: Vercel (frontend static + Python serverless functions)

## Services

| Service | Port | Path | Description |
|---------|------|------|-------------|
| React Frontend | 20446 | `/` | Review input, demo, dashboard |
| Python ML API | 8090 | `/ml` | FastAPI NLP/ML backend |
| Node.js API | 8080 | `/api` | Express health + utility |

## Key Files

- `api/index.py` — Vercel-ready FastAPI ML backend (serverless function)
- `api/requirements.txt` — Python deps for Vercel
- `artifacts/review-api/main.py` — Local development Python ML server
- `artifacts/review-detector/` — React + Vite frontend
- `vercel.json` — Vercel deployment configuration
- `README.md` — Full deployment guide

## ML Endpoints (Python FastAPI)

- `POST /ml/analyze` — Analyze a single review for fake detection
- `POST /ml/batch` — Analyze multiple reviews
- `GET /ml/demo` — Pre-analyzed demo reviews
- `GET /ml/reviews` — All analyzed reviews from DB
- `GET /ml/stats` — Aggregate stats (fake rate, sentiment, top flags)
- `GET /ml/health` — Health check

## ML Features

**Text Features**: TF-IDF bigrams, exclamation density, ALL-CAPS ratio, generic phrase detection, lexical diversity

**Behavioral Features**: New account burst, review frequency, rating extremism, sentiment-vs-rating mismatch

**Explainability**: Per-review NLP feature breakdown + behavioral flag labels

## Deploy to Vercel

1. Push to GitHub
2. Import repo at vercel.com/new — config is auto-detected from `vercel.json`
3. Optionally add `DATABASE_URL` env var for review history persistence

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/review-detector run dev` — run frontend locally
- Python ML API starts automatically via the `ML API` workflow

## User Preferences

- Python for backend ML (over Node.js)
- JavaScript (JSX) for frontend (over TypeScript)
- Vercel deployment via GitHub
