# consumer_complaint_nlp_topic_modeling
NLP pipeline on 70K+ CFPB complaint narratives - LDA topic modeling (coherence-optimized, k=4) + keyword-based UDAAP risk flagging.

## Problem statement
Financial services companies receive large volumes of unstructured customer complaints. Identifying complaints that signal potentially Unfair, Deceptive, or Abusive Acts or Practices (UDAAP) is typically a manual, keyword-based process. This project explores whether unsupervised topic modeling can complement that process — both by automatically surfacing complaint themes without predefined categories, and by checking whether those themes align with known risk signals.

**Dataset:** CFPB Consumer Complaint Database — public, real-world financial services complaint narratives.
https://www.consumerfinance.gov/data-research/consumer-complaints/#get-the-data

## APPROAC
### 1. Preprocessing
Cleaned raw complaint narratives: lowercased, removed redaction placeholders (XXXX), punctuation, and numbers
Tokenized, removed stopwords (standard English + domain-specific terms like "account," "bureau," "report" that appear across nearly all complaints regardless of topic), and lemmatized using NLTK
Final corpus: 70,679 complaints with non-empty narrative text

### 2. Topic modeling
Vectorized using Bag-of-Words (CountVectorizer, max_df=0.95, min_df=2)
Fit LDA models across k = 4 to 7 topics
Selected k using coherence score (c_v, via gensim) to quantitatively justify topic count rather than choosing arbitrarily
k = 4 gave the best coherence score (≈ 0.56–0.58 depending on run)


Resulting topics

Topic | Top words | Label
1 | payment, bank, card, loan, charge, due, paid | Billing & Payment Disputes
2 | consent, authorization, without, third, social | Unauthorized Access & Consent Violations
3 | theft, identity, privacy, violated, fraud | Identity Theft & Privacy Violations
4 | inaccurate, bureau, violation, dispute, remove | Credit Reporting Inaccuracies & FCRA Disputes

### 3.UDAAP risk flagging
Built a keyword-based flagging layer (terms: unfair, deceptive, abusive, misleading, unauthorized, without consent, etc.) — replicating the kind of rule-based detection currently used in practice
17.3% of complaints flagged overall
Cross-tabulated flag rate against topic assignment: "Unauthorized Access & Consent Violations" showed a 60.6% flag rate — roughly 3.5x the baseline


## Findings:
LDA, working purely on word co-occurrence with no knowledge of the UDAAP keyword list, independently surfaced a topic that strongly overlaps with UDAAP-risk language. This is a directionally interesting result — it suggests topic modeling could help prioritize which complaint clusters deserve closer compliance review.

### Note:
The UDAAP keyword list includes terms like "unauthorized," which also appear among that topic's defining LDA words ("authorization," "without"). Some of the correlation is therefore mechanical vocabulary overlap rather than fully independent confirmation. This is a directional finding, not a validated causal result.

**Tech stack:** Python, pandas, NLTK, scikit-learn, gensim

**Next steps:**
1. Compare topic-model-based flagging against the keyword baseline on a labeled subset, if ground truth becomes available
2. Test BERTopic for contextual (embedding-based) topic discovery as a comparison to LDA

Topic 2:
  inquiry | identity | consent | personal | accpasted
