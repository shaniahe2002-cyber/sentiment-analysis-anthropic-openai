# Public Perception of AI Assistants: Anthropic vs. OpenAI

A comparative sentiment analysis of 980 US App Store reviews for Claude and ChatGPT, collected May 1 to June 8, 2026.

## Problem

Anthropic and OpenAI both publish AI assistant apps on the same App Store, but public sentiment toward them is rarely compared directly with real data. This analysis pulls reviews for both apps and applies lexicon based sentiment scoring, emotion classification, and keyword analysis to find where the two brands actually differ in how users talk about them.

## Tools

R 4.6.0. itunesr for data collection, tidytext for tokenization and TF-IDF, the AFINN and NRC lexicons for sentiment and emotion scoring, ggplot2 for visualization.

## Key Findings

- Claude reviews are highly polarized. 46% of scored documents are negative versus 32% for ChatGPT, driven by recurring frustration with usage limits, account actions, and platform specific issues.
- ChatGPT reviews skew more consistently positive at 63%, driven by users describing broad day to day utility rather than specific features.
- TF-IDF analysis shows Claude's user base is more technically specific, referencing model tiers like sonnet and opus, platforms like desktop and ios, and concepts like tokens and coding.
- ChatGPT's user base is broader and more globally distributed, with significant Spanish language content appearing inside English language reviews.

## Methodology

The pipeline runs in four stages. App Store reviews are pulled through the itunesr package for both apps, up to 500 reviews per brand. Text is lowercased, stripped of URLs and punctuation, and filtered to reviews with 5 or more words, since shorter reviews rarely carry enough signal for reliable lexicon scoring. Tokens are then scored with the AFINN lexicon for sentiment polarity and the NRC lexicon for 8 category emotion classification. Finally, unigram frequency, TF-IDF, and bigram analysis surface the terms and phrases that most differentiate the two brands.

## What's in This Repo

| File | Description |
|---|---|
| `sentiment_analysis_anthropic_openai.R` | Full reproducible script, all 4 stages from data collection through visualization |
| `Shaniah_Edwards_OpenAI_Anthropic_Sentiment_Analysis_Report.pdf` | Full case study with methodology, results, and findings |
| `01_polarity_distribution.png` | Sentiment polarity distribution by brand |
| `02_nrc_emotions.png` | NRC emotion profile by brand |
| `03_tfidf_terms.png` | Top brand differentiating terms by TF-IDF |
| `04_rating_distribution.png` | Star rating distribution by brand |

## Skills Demonstrated

R, tidytext, dplyr, ggplot2, lexicon based sentiment analysis, TF-IDF, NRC emotion classification, text mining, data visualization

---

All data was collected directly from the public App Store review feed using the itunesr package. No review content was fabricated or altered.

**Analyst:** Shaniah Edwards | MBA, Business Analytics
