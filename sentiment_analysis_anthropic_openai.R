# ============================================================
# Sentiment Analysis: Anthropic (Claude) vs OpenAI (ChatGPT)
# App Store Reviews | May-June 8 2026
# Analyst: Shaniah Edwards | MBA Business Analytics
# ============================================================

# ── LIBRARIES ────────────────────────────────────────────────
library(itunesr)
library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(syuzhet)
library(ggplot2)
library(wordcloud)
library(reshape2)
library(lubridate)
library(scales)
library(textdata)

# ── 1. DATA COLLECTION ───────────────────────────────────────
# Pull up to 500 reviews per app from the US App Store
# Claude App ID: 6473753684 | ChatGPT App ID: 6448311069

pull_reviews <- function(app_id, pages = 10) {
  bind_rows(lapply(1:pages, function(p) {
    tryCatch(getReviews(app_id, "us", p), error = function(e) NULL)
  }))
}

claude_raw  <- pull_reviews(6473753684)
chatgpt_raw <- pull_reviews(6448311069)

claude_raw$brand  <- "Anthropic"
chatgpt_raw$brand <- "OpenAI"

corpus <- bind_rows(claude_raw, chatgpt_raw)
write.csv(corpus, "app_store_corpus.csv", row.names = FALSE)

cat("Records collected:", nrow(corpus), "\n")
cat("Anthropic:", sum(corpus$brand == "Anthropic"), "\n")
cat("OpenAI:", sum(corpus$brand == "OpenAI"), "\n")

glimpse(claude_raw)
glimpse(chatgpt_raw)

# ── 2. PREPROCESSING ─────────────────────────────────────────
corpus_clean <- corpus |>
  mutate(
    text_clean = Review |>
      str_to_lower() |>
      str_remove_all("https?://[^\\s]+") |>
      str_remove_all("[^a-z0-9\\s]") |>
      str_squish(),
    date_parsed = as.Date(Date),
    month = floor_date(date_parsed, "month"),
    Rating = as.numeric(Rating),
    doc_id = row_number()
  ) |>
  filter(str_count(text_clean, "\\w+") >= 5)

cat("After cleaning:", nrow(corpus_clean), "documents\n")

# Custom stopwords to remove domain noise
custom_stops <- tibble(
  word = c("ai", "model", "use", "just", "like", "claude", "chatgpt",
           "gpt", "anthropic", "openai", "app", "chat", "it's", "i'm",
           "don't", "can't", "even", "really", "get", "got", "one",
           "also", "still", "now", "made", "make", "used", "using"),
  lexicon = "custom"
)

all_stops <- bind_rows(stop_words, custom_stops)

# Tokenize and remove stopwords
tokens <- corpus_clean |>
  unnest_tokens(word, text_clean) |>
  anti_join(all_stops, by = "word") |>
  filter(str_length(word) > 2)

cat("Total tokens after cleaning:", nrow(tokens), "\n")

# ── 3. SENTIMENT SCORING ─────────────────────────────────────
# AFINN: numeric valence scoring (-5 to +5)
afinn_scores <- tokens |>
  inner_join(get_sentiments("afinn"), by = "word") |>
  group_by(doc_id, brand, Rating, month) |>
  summarise(
    sentiment_score = sum(value),
    word_count = n(),
    normalized_score = sentiment_score / word_count,
    .groups = "drop"
  ) |>
  mutate(
    polarity = case_when(
      normalized_score > 0.1  ~ "Positive",
      normalized_score < -0.1 ~ "Negative",
      TRUE                    ~ "Neutral"
    )
  )

cat("AFINN scored documents:", nrow(afinn_scores), "\n")
print(table(afinn_scores$polarity, afinn_scores$brand))

# NRC: emotion classification (8 emotions)
nrc_scores <- tokens |>
  inner_join(get_sentiments("nrc"), by = "word") |>
  filter(!sentiment %in% c("positive", "negative")) |>
  count(brand, sentiment) |>
  group_by(brand) |>
  mutate(pct = n / sum(n))

# ── 4. TEXT MINING ───────────────────────────────────────────
# Top unigrams by brand
top_terms <- tokens |>
  count(brand, word, sort = TRUE) |>
  group_by(brand) |>
  slice_max(n, n = 15) |>
  ungroup()

# TF-IDF: brand-differentiating terms
tfidf <- tokens |>
  count(brand, word) |>
  bind_tf_idf(word, brand, n) |>
  group_by(brand) |>
  slice_max(tf_idf, n = 15) |>
  ungroup()

# Bigrams: two-word phrases
bigrams <- corpus_clean |>
  unnest_tokens(bigram, text_clean, token = "ngrams", n = 2) |>
  separate(bigram, c("word1", "word2"), sep = " ") |>
  filter(
    !word1 %in% all_stops$word,
    !word2 %in% all_stops$word,
    str_length(word1) > 2,
    str_length(word2) > 2
  ) |>
  unite(bigram, word1, word2, sep = " ") |>
  count(brand, bigram, sort = TRUE) |>
  group_by(brand) |>
  slice_max(n, n = 10) |>
  ungroup()

# Monthly sentiment summary
monthly <- afinn_scores |>
  group_by(brand, month) |>
  summarise(
    mean_score = mean(normalized_score, na.rm = TRUE),
    record_count = n(),
    pct_positive = mean(polarity == "Positive"),
    .groups = "drop"
  )

# ── 5. VISUALIZATIONS ────────────────────────────────────────
# Figure 1: Sentiment Polarity Distribution
polarity_summary <- afinn_scores |>
  count(brand, polarity) |>
  group_by(brand) |>
  mutate(pct = n / sum(n))

# Figure 1: Sentiment Polarity Distribution
p1 <- ggplot(polarity_summary, aes(x = polarity, y = pct, fill = brand)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_text(
    aes(label = paste0(round(pct * 100, 1), "%")),
    position = position_dodge(width = 0.6),
    vjust = -0.4,
    size = 3.5,
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c("Anthropic" = "#C4622D", "OpenAI" = "#10A37F"),
    labels = c("Anthropic" = "Anthropic (Claude)", "OpenAI" = "OpenAI (ChatGPT)")
  ) +
  scale_y_continuous(
    labels = percent_format(),
    limits = c(0, 0.80),
    expand = c(0, 0)
  ) +
  labs(
    title = "Sentiment Polarity Distribution by Brand",
    subtitle = "App Store Reviews, May 1 - June 8 2026  |  n = 559 scored documents",
    x = NULL, y = "% of Scored Reviews", fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "top",
    legend.key.size = unit(0.4, "cm"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = "grey80", linewidth = 0.5)
  )
print(p1)

# Figure 2: NRC Emotion Profile
p2 <- ggplot(nrc_scores, aes(x = reorder(sentiment, pct), y = pct, fill = brand)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_text(
    aes(label = paste0(round(pct * 100, 1), "%")),
    position = position_dodge(width = 0.6),
    hjust = -0.15,
    size = 3,
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c("Anthropic" = "#C4622D", "OpenAI" = "#10A37F"),
    labels = c("Anthropic" = "Anthropic (Claude)", "OpenAI" = "OpenAI (ChatGPT)")
  ) +
  scale_y_continuous(
    labels = percent_format(),
    limits = c(0, 0.35),
    expand = c(0, 0)
  ) +
  coord_flip() +
  labs(
    title = "NRC Emotion Profile by Brand",
    subtitle = "% of emotion-classified tokens  |  App Store Reviews May 1 - June 8 2026",
    x = NULL, y = "% of Classified Tokens", fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "top",
    legend.key.size = unit(0.4, "cm"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = "grey80", linewidth = 0.5)
  )
print(p2)

# Figure 3: TF-IDF Brand-Differentiating Terms
p3 <- tfidf |>
  group_by(brand) |>
  slice_max(tf_idf, n = 10) |>
  ggplot(aes(x = reorder_within(word, tf_idf, brand), y = tf_idf, fill = brand)) +
  geom_col(show.legend = FALSE) +
  geom_text(
    aes(label = round(tf_idf, 4)),
    hjust = -0.15,
    size = 2.8,
    fontface = "bold"
  ) +
  scale_fill_manual(values = c("Anthropic" = "#C4622D", "OpenAI" = "#10A37F")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  facet_wrap(~brand, scales = "free") +
  scale_x_reordered() +
  coord_flip() +
  labs(
    title = "Top Brand-Differentiating Terms by TF-IDF",
    subtitle = "Terms weighted by uniqueness to each brand",
    x = NULL, y = "TF-IDF Score"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    strip.text = element_text(face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = "grey80", linewidth = 0.5)
  )
print(p3)

# Figure 4: Star Rating Distribution
p4 <- corpus_clean |>
  mutate(Rating = as.numeric(Rating)) |>
  count(brand, Rating) |>
  group_by(brand) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = factor(Rating), y = pct, fill = brand)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_text(
    aes(label = paste0(round(pct * 100, 1), "%")),
    position = position_dodge(width = 0.6),
    vjust = -0.4,
    size = 3.5,
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c("Anthropic" = "#C4622D", "OpenAI" = "#10A37F"),
    labels = c("Anthropic" = "Anthropic (Claude)", "OpenAI" = "OpenAI (ChatGPT)")
  ) +
  scale_y_continuous(
    labels = percent_format(),
    limits = c(0, 0.80),
    expand = c(0, 0)
  ) +
  labs(
    title = "Star Rating Distribution by Brand",
    subtitle = "App Store Reviews  |  May 1 - June 8 2026  |  n = 782",
    x = "Star Rating", y = "% of Reviews", fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "top",
    legend.key.size = unit(0.4, "cm"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = "grey80", linewidth = 0.5)
  )
print(p4)

# ── 6. SAVE OUTPUTS ──────────────────────────────────────────
ggsave("01_polarity_distribution.png", plot = p1, width = 8, height = 5, dpi = 150)
ggsave("02_nrc_emotions.png",          plot = p2, width = 8, height = 5, dpi = 150)
ggsave("03_tfidf_terms.png",           plot = p3, width = 10, height = 6, dpi = 150)
ggsave("04_rating_distribution.png",   plot = p4, width = 8, height = 5, dpi = 150)

save(corpus_clean, tokens, afinn_scores, nrc_scores,
     top_terms, tfidf, bigrams, monthly,
     file = "sentiment_analysis.RData")

cat("All outputs saved to", getwd(), "\n")
cat("Analysis complete.\n")

# ── 7. SESSION INFO ──────────────────────────────────────────
# Recorded for reproducibility
sessionInfo()

