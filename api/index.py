import os
import re
import json
import string
import logging
import psycopg2
import psycopg2.extras
import nltk
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
from typing import Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data at cold start
nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Training corpus (synthetic — model trains at cold start, ~100ms)
# ---------------------------------------------------------------------------
FAKE_REVIEWS = [
    "This product is absolutely amazing! Best purchase ever! Five stars!",
    "WOW!! I can't believe how great this is! AMAZING QUALITY!!",
    "Perfect product. Perfect shipping. Perfect everything. 5 stars!",
    "This changed my life completely! Everyone should buy this NOW!",
    "Incredible value!! So happy with my purchase! Highly recommended!!!",
    "Best product on the market hands down. No complaints whatsoever!",
    "I bought this for my whole family. Everyone loves it so much!!!",
    "This exceeded all my expectations! Just buy it trust me 100%!",
    "Absolutely flawless product. Zero defects. Perfect in every way!",
    "I've tried many products and this is BY FAR the absolute best!!",
    "LOVE LOVE LOVE this product!!! Will buy again and again!",
    "Just received today and already obsessed. Best thing I've bought.",
    "Do yourself a favor and buy this. You will not regret it at all!",
    "Outstanding quality, outstanding price, outstanding everything!!",
    "This is pure perfection. Nothing bad to say. Buy it immediately!",
    "Wow just wow. This product is on another level. Truly amazing!!!",
    "I showed this to all my friends and they all want one too!!",
    "Cannot be happier with this! Perfect in every single way possible!",
    "100% worth every penny and more! Exceeded ALL my expectations!",
    "I am so in love with this product!! Best purchase of my entire life!",
    "Amazing product works exactly as advertised. Highly recommend!",
    "This product is phenomenal. Shipping was fast. Packaging great!",
    "Would give 10 stars if I could. Absolutely love love love it!!!",
    "Bought 3 of these already! That says it all about the quality!!",
]

REAL_REVIEWS = [
    "The product works well for the most part. Delivery was a bit slow.",
    "Pretty good quality. Some minor issues but nothing too serious.",
    "Does what it says on the tin. Nothing spectacular but reliable.",
    "Mixed feelings. The build quality is okay but the design is awkward.",
    "Decent product overall. Had to return the first one due to a defect.",
    "Works as expected. Setup instructions could be clearer though.",
    "Good value for money but don't expect premium quality at this price.",
    "The battery life is shorter than advertised but otherwise fine.",
    "It's okay. Got the job done for a few months then started having issues.",
    "Shipping was delayed by a week but the product itself is satisfactory.",
    "I like it but my partner isn't as impressed. Color looks different in person.",
    "Three months in and still working. Hope it continues to hold up.",
    "Solid product. Assembly was frustrating but end result looks good.",
    "Recommended by a friend. It's decent, not life changing but useful.",
    "Arrived slightly damaged but customer service sorted it out quickly.",
    "Better than what I had before. Not perfect but does the job well.",
    "Good enough for the price. Wouldn't pay more for it though.",
    "The first batch I received had issues. The replacement is working fine.",
    "Works well in warm weather. Haven't tested in winter yet.",
    "Comfortable and practical. Wish it came in more color options.",
    "Quality is inconsistent. Some parts feel sturdy, others feel cheap.",
    "Used it for a week now. So far so good but ask me again in a month.",
    "Fits my needs. Read other reviews before buying so had realistic expectations.",
    "Decent but nothing special. Does the basic job without any fuss.",
]

GENERIC_FAKE_PHRASES = [
    "best purchase ever", "changed my life", "absolutely amazing", "highly recommend",
    "exceeded expectations", "five stars", "perfect in every way", "do yourself a favor",
    "buy it immediately", "worth every penny", "not regret", "trust me",
    "best thing", "everyone should buy", "just buy it", "love love love",
    "best product ever", "absolutely flawless", "pure perfection",
]

DEMO_REVIEWS = [
    {
        "id": 1, "reviewer_id": "user_001",
        "product": "Wireless Earbuds Pro X", "rating": 5,
        "text": "WOW!! These earbuds are absolutely AMAZING!! Best purchase I've ever made in my entire life!! Sound quality is PERFECT!! Everyone should buy these immediately! I showed all my friends and family and they all want a pair now!! 100% recommend!! Changed my life completely!!",
        "helpful_votes": 2,
        "reviewer_history": {"total_reviews": 3, "avg_rating": 5.0, "days_since_joined": 5, "reviews_in_30_days": 3},
    },
    {
        "id": 2, "reviewer_id": "user_002",
        "product": "Wireless Earbuds Pro X", "rating": 3,
        "text": "Sound quality is decent for the price. Bass is a bit weak compared to my previous pair. Battery lasts about 5 hours which matches the listing. Fit is comfortable but the right earbud feels slightly loose. Returned the first pair due to connection drops, replacement is working fine so far.",
        "helpful_votes": 47,
        "reviewer_history": {"total_reviews": 89, "avg_rating": 3.6, "days_since_joined": 1200, "reviews_in_30_days": 2},
    },
    {
        "id": 3, "reviewer_id": "user_003",
        "product": "Kitchen Knife Set", "rating": 5,
        "text": "Perfect knives! Perfect quality! Perfect everything! Best kitchen purchase EVER!! I can't believe how amazing these are! Would give 10 stars if possible! Buy now you will not regret!!!",
        "helpful_votes": 1,
        "reviewer_history": {"total_reviews": 2, "avg_rating": 5.0, "days_since_joined": 3, "reviews_in_30_days": 2},
    },
    {
        "id": 4, "reviewer_id": "user_004",
        "product": "Kitchen Knife Set", "rating": 4,
        "text": "The chef's knife is excellent — sharp out of the box and holds an edge well after 3 months of regular use. The paring knife is a bit flimsy. Handles feel comfortable for most tasks. Would have preferred a honing steel included at this price point.",
        "helpful_votes": 63,
        "reviewer_history": {"total_reviews": 156, "avg_rating": 3.8, "days_since_joined": 2400, "reviews_in_30_days": 1},
    },
    {
        "id": 5, "reviewer_id": "user_005",
        "product": "Yoga Mat Premium", "rating": 1,
        "text": "TERRIBLE!! This mat is absolutely the WORST thing I've ever bought!! Complete garbage!! Do not buy this under any circumstances!! Worst company ever!!",
        "helpful_votes": 0,
        "reviewer_history": {"total_reviews": 4, "avg_rating": 1.0, "days_since_joined": 7, "reviews_in_30_days": 4},
    },
    {
        "id": 6, "reviewer_id": "user_006",
        "product": "Yoga Mat Premium", "rating": 4,
        "text": "Good mat overall. Non-slip surface works well on hardwood floors, less so on carpet. Thickness is adequate for joint support. Slight chemical smell when first opened that faded after airing out for two days. Bag is convenient. At this price it's good value.",
        "helpful_votes": 38,
        "reviewer_history": {"total_reviews": 44, "avg_rating": 3.9, "days_since_joined": 890, "reviews_in_30_days": 1},
    },
]

# ---------------------------------------------------------------------------
# Train ML model at module load (fast — synthetic data only)
# ---------------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()

_texts = FAKE_REVIEWS + REAL_REVIEWS
_labels = [1] * len(FAKE_REVIEWS) + [0] * len(REAL_REVIEWS)

nlp_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, min_df=1,
                              strip_accents="unicode", analyzer="word")),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
])
nlp_model.fit(_texts, _labels)
logger.info("ML model trained")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    try:
        return psycopg2.connect(url)
    except Exception as e:
        logger.warning(f"DB connection failed: {e}")
        return None


def init_db():
    conn = get_db()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analyzed_reviews (
                id SERIAL PRIMARY KEY,
                reviewer_id TEXT,
                product TEXT,
                rating INTEGER,
                review_text TEXT NOT NULL,
                is_fake BOOLEAN NOT NULL,
                fake_score FLOAT NOT NULL,
                confidence TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                sentiment_score FLOAT NOT NULL,
                sentiment_mismatch BOOLEAN NOT NULL DEFAULT FALSE,
                behavioral_flags JSONB DEFAULT '[]',
                nlp_features JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
    except Exception as e:
        logger.warning(f"DB init error: {e}")
    finally:
        conn.close()


init_db()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ReviewerHistory(BaseModel):
    total_reviews: Optional[int] = None
    avg_rating: Optional[float] = None
    days_since_joined: Optional[int] = None
    reviews_in_30_days: Optional[int] = None


class ReviewRequest(BaseModel):
    review_text: str
    rating: Optional[int] = None
    reviewer_id: Optional[str] = None
    product: Optional[str] = None
    reviewer_history: Optional[ReviewerHistory] = None


class BatchReviewRequest(BaseModel):
    reviews: List[ReviewRequest]

# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def extract_nlp_features(text: str):
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    exclamation_count = text.count("!")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
    avg_word_len = sum(len(w.strip(string.punctuation)) for w in words) / max(word_count, 1)
    generic_hits = sum(1 for p in GENERIC_FAKE_PHRASES if p.lower() in text.lower())
    unique_words = len(set(w.lower().strip(string.punctuation) for w in words))
    lexical_diversity = unique_words / max(word_count, 1)
    repeated = sum(1 for p in ["love love", "amazing amazing", "best best", "perfect perfect"]
                   if p in text.lower())
    return {
        "word_count": word_count,
        "exclamation_count": exclamation_count,
        "question_count": text.count("?"),
        "caps_ratio": round(caps_ratio, 3),
        "avg_word_length": round(avg_word_len, 2),
        "generic_phrase_hits": generic_hits,
        "lexical_diversity": round(lexical_diversity, 3),
        "repeated_phrases": repeated,
    }


def analyze_sentiment(text: str, rating: Optional[int] = None):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.5:
        sentiment, score = "positive", compound
    elif compound <= -0.5:
        sentiment, score = "negative", compound
    else:
        sentiment, score = "neutral", compound

    mismatch = False
    if rating is not None:
        if (rating >= 4 and compound < 0.0) or (rating <= 2 and compound > 0.2):
            mismatch = True
        elif rating == 5 and compound < 0.5:
            mismatch = True
        elif rating == 1 and compound > -0.3:
            mismatch = True
    return sentiment, round(score, 3), mismatch


def analyze_behavior(history: Optional[ReviewerHistory], rating: Optional[int], features: dict):
    flags = []
    if history:
        if (history.total_reviews or 0) >= 2 and (history.days_since_joined or 999) <= 7:
            flags.append("new_account_burst")
        if (history.reviews_in_30_days or 0) >= 5:
            flags.append("high_review_frequency")
        if history.avg_rating == 5.0 and (history.total_reviews or 0) > 3:
            flags.append("rating_extremism_high")
        elif (history.avg_rating or 5) <= 1.5 and (history.total_reviews or 0) > 3:
            flags.append("rating_extremism_low")
    if features["exclamation_count"] >= 4:
        flags.append("excessive_punctuation")
    if features["caps_ratio"] > 0.25:
        flags.append("excessive_caps")
    if features["generic_phrase_hits"] >= 2:
        flags.append("generic_language")
    if features["word_count"] < 10:
        flags.append("suspiciously_short")
    if features["lexical_diversity"] < 0.4 and features["word_count"] > 15:
        flags.append("low_lexical_diversity")
    if features["repeated_phrases"] > 0:
        flags.append("repeated_phrases")
    if rating == 5 and features["generic_phrase_hits"] >= 1 and features["exclamation_count"] >= 2:
        flags.append("suspiciously_enthusiastic")
    return list(set(flags))


def classify_review(text: str, rating: Optional[int], history: Optional[ReviewerHistory]):
    nlp_prob = float(nlp_model.predict_proba([text])[0][1])
    features = extract_nlp_features(text)
    sentiment, sentiment_score, mismatch = analyze_sentiment(text, rating)
    flags = analyze_behavior(history, rating, features)

    score = nlp_prob
    score += 0.05 * min(features["exclamation_count"], 6) if features["exclamation_count"] >= 3 else 0
    score += 0.08 if features["caps_ratio"] > 0.2 else 0
    score += 0.06 * features["generic_phrase_hits"] if features["generic_phrase_hits"] >= 2 else 0
    score += 0.12 if mismatch else 0
    score += 0.18 if "new_account_burst" in flags else 0
    score += 0.10 if "high_review_frequency" in flags else 0
    score += 0.10 if any(f in flags for f in ("rating_extremism_high", "rating_extremism_low")) else 0
    score += 0.08 if "suspiciously_enthusiastic" in flags else 0
    score += 0.05 if features["word_count"] < 10 else 0
    score += 0.07 if features["lexical_diversity"] < 0.4 and features["word_count"] > 15 else 0

    fake_score = round(min(max(score, 0.01), 0.99), 3)
    is_fake = fake_score >= 0.55
    if fake_score >= 0.80 or fake_score <= 0.25:
        confidence = "high"
    elif fake_score >= 0.60 or fake_score <= 0.40:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "is_fake": is_fake,
        "fake_score": fake_score,
        "confidence": confidence,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "sentiment_mismatch": mismatch,
        "behavioral_flags": flags,
        "nlp_features": features,
    }

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Fake Review Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ml/health")
@app.get("/api/health")
def health():
    return {"status": "ok", "model": "trained"}


@app.post("/ml/analyze")
@app.post("/api/analyze")
def analyze_review(req: ReviewRequest):
    if not req.review_text or len(req.review_text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Review text too short")

    result = classify_review(req.review_text, req.rating, req.reviewer_history)
    record = {"reviewer_id": req.reviewer_id, "product": req.product,
              "rating": req.rating, "review_text": req.review_text, **result}

    conn = get_db()
    saved_id = None
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO analyzed_reviews
                   (reviewer_id, product, rating, review_text, is_fake, fake_score,
                    confidence, sentiment, sentiment_score, sentiment_mismatch,
                    behavioral_flags, nlp_features)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (req.reviewer_id, req.product, req.rating, req.review_text,
                 result["is_fake"], result["fake_score"], result["confidence"],
                 result["sentiment"], result["sentiment_score"], result["sentiment_mismatch"],
                 json.dumps(result["behavioral_flags"]), json.dumps(result["nlp_features"]))
            )
            saved_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
        except Exception as e:
            logger.warning(f"DB insert error: {e}")
        finally:
            conn.close()

    return {**record, "id": saved_id}


@app.post("/ml/batch")
@app.post("/api/batch")
def analyze_batch(req: BatchReviewRequest):
    results = []
    for r in req.reviews:
        res = classify_review(r.review_text, r.rating, r.reviewer_history)
        results.append({"reviewer_id": r.reviewer_id, "product": r.product,
                        "rating": r.rating, "review_text": r.review_text, **res})
    return {"results": results, "total": len(results)}


@app.get("/ml/reviews")
@app.get("/api/reviews")
def get_reviews(limit: int = 50, offset: int = 0):
    conn = get_db()
    if not conn:
        return {"reviews": [], "total": 0}
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT COUNT(*) as count FROM analyzed_reviews")
        total = cur.fetchone()["count"]
        cur.execute("SELECT * FROM analyzed_reviews ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (limit, offset))
        rows = cur.fetchall()
        cur.close()
        return {"reviews": [dict(r) for r in rows], "total": total}
    except Exception as e:
        logger.warning(f"DB error: {e}")
        return {"reviews": [], "total": 0}
    finally:
        conn.close()


@app.get("/ml/stats")
@app.get("/api/stats")
def get_stats():
    conn = get_db()
    if not conn:
        return {"total_analyzed": 0, "fake_count": 0, "real_count": 0,
                "fake_rate": 0, "avg_fake_score": 0,
                "sentiment_breakdown": {"positive": 0, "negative": 0, "neutral": 0},
                "top_flags": []}
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN is_fake THEN 1 ELSE 0 END) as fake_count,
                   AVG(fake_score) as avg_fake_score,
                   SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) as positive,
                   SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) as negative,
                   SUM(CASE WHEN sentiment='neutral' THEN 1 ELSE 0 END) as neutral
            FROM analyzed_reviews
        """)
        row = dict(cur.fetchone())
        cur.execute("SELECT behavioral_flags FROM analyzed_reviews WHERE behavioral_flags != '[]'::jsonb")
        flag_rows = cur.fetchall()
        cur.close()

        flag_counts = {}
        for fr in flag_rows:
            for f in (fr["behavioral_flags"] or []):
                flag_counts[f] = flag_counts.get(f, 0) + 1
        top_flags = [{"flag": f, "count": c} for f, c in
                     sorted(flag_counts.items(), key=lambda x: -x[1])[:5]]

        total = int(row["total"] or 0)
        fake = int(row["fake_count"] or 0)
        return {
            "total_analyzed": total, "fake_count": fake, "real_count": total - fake,
            "fake_rate": round(fake / total, 3) if total > 0 else 0,
            "avg_fake_score": round(float(row["avg_fake_score"] or 0), 3),
            "sentiment_breakdown": {"positive": int(row["positive"] or 0),
                                    "negative": int(row["negative"] or 0),
                                    "neutral": int(row["neutral"] or 0)},
            "top_flags": top_flags,
        }
    except Exception as e:
        logger.warning(f"Stats error: {e}")
        return {"total_analyzed": 0, "fake_count": 0, "real_count": 0,
                "fake_rate": 0, "avg_fake_score": 0}
    finally:
        conn.close()


@app.get("/ml/demo")
@app.get("/api/demo")
def get_demo():
    results = []
    for r in DEMO_REVIEWS:
        history = ReviewerHistory(**r["reviewer_history"]) if "reviewer_history" in r else None
        res = classify_review(r["text"], r["rating"], history)
        results.append({**r, **res})
    return {"reviews": results}


# Vercel ASGI handler
handler = Mangum(app, lifespan="off")
