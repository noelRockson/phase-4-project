# NLP Sentiment Analysis: Apple vs Google Tweets (Non-Technical)

## 1) What we set out to do
- Objective: Understand how people feel about Apple and Google by classifying tweets as negative, neutral, or positive.
- Why it matters: Helps marketing and product teams monitor brand reputation, spot issues quickly, and measure the impact of campaigns.

## 2) The data we used
- Source: A labeled dataset of tweets mentioning Apple/Google with three sentiment labels: negative, neutral, positive.
- Size and balance: The data is skewed toward neutral and positive; negative tweets are relatively rare.
- Basic cleaning: We removed links, user mentions, hashtags, punctuation, and lowercased text. Common words (like "the", "and") were filtered out so the model focuses on more meaningful words.

## 3) How the model “reads” text (in plain terms)
- We convert text into a numeric representation using TF-IDF. Think of this as measuring how distinctive each word is across the dataset.
- We trained a Logistic Regression classifier, a well-established method for text classification that balances performance and interpretability.
- We built two views:
  - Binary model: positive vs. negative (for a focused comparison).
  - Multiclass model: negative, neutral, positive (the real task).

## 4) Results and what they mean
- We evaluated accuracy (overall correctness) and per-class precision/recall/F1.
- Key takeaways (from typical outcomes in our notebook):
  - Positive and neutral tweets are easier for the model; negative is harder due to fewer examples and overlapping language.
  - Errors often occur between neutral and positive, and between negative and neutral (e.g., mild criticism reads like neutral without context).
  - In a business context, if catching negative sentiment is the priority, we can adjust the model to favor recall on the negative class (i.e., miss fewer negatives), even if that slightly increases false alarms.

### Visuals
![Class distribution](figures/class_distribution_pie.png)

![Binary confusion matrix](figures/confusion_binary.png)

![Multiclass confusion matrix](figures/confusion_multiclass.png)

![Multiclass metrics](figures/metrics_multiclass.png)

### How to read the metrics (non-technical)
- Accuracy: % of all tweets classified correctly.
- Precision (for a class): Of the tweets the model said were, e.g., "negative", how many truly were negative? (False alarms hurt precision.)
- Recall (for a class): Of all truly negative tweets, how many did the model catch? (Missed negatives hurt recall.)
- F1-score: A balance of precision and recall — useful when classes are imbalanced.

### Confusion matrix (at a glance)
- A confusion matrix is a simple table comparing model predictions vs. actual labels.
- More values on the diagonal (top-left to bottom-right) = better performance.
- Off-diagonal cells show specific confusions (e.g., negative → neutral), which guide targeted improvements.

## 5) Feature insights (why the model predicts the way it does)
- With Logistic Regression, each word gets a weight per class:
  - Positive weight = pushes a tweet toward that class.
  - Negative weight = pulls a tweet away from that class.
- We reviewed top words per class to validate that learned signals make sense and to spot potential issues (e.g., words tied to news or slang that change over time).

## 6) What to do with these insights
- Monitoring: Build alerts for spikes in negative sentiment and track trends by product, campaign, or geography.
- Customer feedback loop: Route the most confident negative tweets to support/PR for quick follow-up.
- Campaign measurement: Compare sentiment before/after launches; monitor competitor mentions for opportunities.

## 7) Limitations
- Imbalanced data: Fewer negative examples make negative detection harder.
- Nuance and context: Sarcasm, irony, and evolving slang remain challenging.
- Domain drift: Vocabulary and topics change; models need periodic updates.

## 8) How we can improve
- Data: Collect more negative examples; label recent tweets to keep up with new phrasing.
- Features: Add bigrams (two-word phrases), character n-grams, emojis/punctuation features; consider lemmatization.
- Models: Compare with Linear SVM, Complement Naïve Bayes, or a lightweight transformer (e.g., DistilBERT) for potentially better nuance.
- Thresholds: Tune decision thresholds to prioritize what matters most (e.g., catching more negatives).

## 9) Summary for stakeholders
- We can automatically classify tweets into negative/neutral/positive to take the pulse of brand perception.
- Today’s model performs well overall, especially for positive and neutral, and provides a fast, scalable view of public sentiment.
- Focused improvements (more negative examples, better features, threshold tuning) will increase sensitivity to negative tweets — the most actionable signal.

## 10) Appendix (light technical glossary)
- TF-IDF: A way to represent text by how unique words are; common words are down-weighted.
- Logistic Regression: A widely used, interpretable classifier for text.
- Precision/Recall/F1: Measures that explain correctness and coverage per class (see section 4).
- Confusion Matrix: A table that shows where the model confuses classes to guide improvements.

---



