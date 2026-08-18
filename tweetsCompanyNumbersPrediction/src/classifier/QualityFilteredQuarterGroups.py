"""Target-independent quality filtering for company/financial-event tweet groups."""

import re

import pandas as pd

from classifier.PureTextQuarterViews import FINANCE_EVENT_TERMS, normalize_semantic_text
from classifier.QuarterAlignedDataset import reporting_quarters


COMPANY_EVENT_TERMS = {
    "amazon": {
        "amazon", "amzn", "aws", "prime", "retail", "cloud", "orders", "revenue",
        "sales", "earnings", "profit", "margin", "guidance", "demand",
    },
    "apple": {
        "apple", "aapl", "iphone", "ipad", "mac", "services", "appstore", "earnings",
        "eps", "revenue", "sales", "profit", "margin", "guidance", "demand",
    },
    "tesla": {
        "tesla", "tsla", "model", "cars", "vehicles", "deliveries", "delivered",
        "production", "units", "sales", "orders", "demand", "revenue", "margin",
    },
}

QUALITY_EVENT_TERMS = frozenset(FINANCE_EVENT_TERMS).union({
    "business", "customers", "customer", "contract", "contracts", "launch", "launched",
    "factory", "capacity", "supply", "shortage", "inventory", "subscription", "services",
})

PROMOTION_PATTERNS = (
    r"binary\s+options?",
    r"penny\s+stocks?",
    r"profit\s+potential",
    r"option\s+millionaires?",
    r"free\s+(?:stock\s+)?alerts?",
    r"subscribe\s+(?:today|now)",
    r"sign\s*up\s+(?:today|now|free)",
    r"forextrading\s+bonus",
    r"join\s+(?:our|the)\s+(?:trading|signals?)",
)


def _term_pattern(terms):
    return r"\b(?:%s)\b" % "|".join(
        re.escape(term) for term in sorted(terms, key=len, reverse=True))


def quality_filter_tweets(dataframe, company, max_author_tweets_per_quarter=50,
                          minimum_semantic_tokens=4):
    """Keep unique financial/company event tweets and cap repeated author influence."""
    if company not in COMPANY_EVENT_TERMS:
        raise ValueError("Unknown company %s" % company)
    required = {"writer", "post_date", "body", "class"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError("Missing quality-filter columns: %s" % sorted(missing))
    frame = dataframe.copy()
    frame["body"] = frame["body"].fillna("").astype(str)
    frame["writer"] = frame["writer"].fillna("<missing>").astype(str)
    frame["reporting_quarter"] = reporting_quarters(frame["post_date"])
    lower = frame["body"].str.lower()
    event_pattern = _term_pattern(QUALITY_EVENT_TERMS)
    company_pattern = _term_pattern(COMPANY_EVENT_TERMS[company])
    event_mask = lower.str.contains(event_pattern, regex=True, na=False)
    company_mask = lower.str.contains(company_pattern, regex=True, na=False)
    # Strong financial language is sufficient; broader event language must mention the company.
    strong_financial_pattern = _term_pattern({
        "earnings", "eps", "revenue", "sales", "deliveries", "delivered", "production",
        "units", "profit", "margin", "guidance", "forecast", "estimates", "demand",
    })
    strong_financial = lower.str.contains(strong_financial_pattern, regex=True, na=False)
    promotion = lower.str.contains(
        "|".join("(?:%s)" % pattern for pattern in PROMOTION_PATTERNS),
        regex=True,
        na=False,
    )
    frame = frame[(event_mask & company_mask | strong_financial) & ~promotion].copy()
    frame["semantic_text"] = frame["body"].map(normalize_semantic_text)
    token_counts = frame["semantic_text"].str.split().map(len)
    frame = frame[token_counts >= int(minimum_semantic_tokens)].copy()
    frame.sort_values("post_date", kind="stable", inplace=True)
    frame.drop_duplicates(
        subset=["reporting_quarter", "semantic_text"], keep="first", inplace=True)
    author_position = frame.groupby(
        ["reporting_quarter", "writer"], sort=False).cumcount()
    frame = frame[author_position < int(max_author_tweets_per_quarter)].copy()
    frame.reset_index(drop=True, inplace=True)
    return frame
