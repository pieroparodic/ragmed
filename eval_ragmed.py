"""
RAGMed Evaluation Script
========================
Uses Gemini as an LLM-as-a-judge to score the relevance of PubMed articles
retrieved by the RAGMed pipeline.

Metrics computed:
  - Mean relevance score per query (1–5 scale)
  - Overall mean relevance score
  - % of results scoring >= 3 (relevant)
  - % of results scoring >= 4 (highly relevant)
  - Domain accuracy (keyword-based check)
  - Per-query breakdown with Gemini reasoning

Output: eval_results.json  (raw data)
"""

import os
import sys
import json
import time
import re
from datetime import datetime

from google import genai
from google.genai import types

# Add ragmed directory to path
sys.path.insert(0, os.path.dirname(__file__))
from rag_pubmed_v2 import answer_with_pubmed_rag_v2, SemanticReranker

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA4d9kLv4uLXoJVII6K8jHNiDvlIlOCRSY")
GEMINI_MODEL   = "gemini-2.5-flash"

# How many PubMed candidates to fetch per query
PUBMED_RETMAX = 20
# How many top results to evaluate per query
TOP_K = 3

# ──────────────────────────────────────────────
# TEST QUERIES  (question + expected domain)
# ──────────────────────────────────────────────
TEST_QUERIES = [
    # Cardiology
    {"question": "What are the most effective treatments for heart failure with reduced ejection fraction?",
     "domain": "cardiology",
     "domain_keywords": ["heart", "cardiac", "cardiovascular", "ejection", "ventricular"]},

    {"question": "What is the role of statins in preventing cardiovascular events?",
     "domain": "cardiology",
     "domain_keywords": ["statin", "cardiovascular", "cholesterol", "lipid", "cardiac"]},

    # Oncology
    {"question": "How effective is immunotherapy for non-small cell lung cancer?",
     "domain": "oncology",
     "domain_keywords": ["cancer", "tumor", "immunotherapy", "lung", "oncology"]},

    {"question": "What are the side effects of chemotherapy in breast cancer patients?",
     "domain": "oncology",
     "domain_keywords": ["cancer", "chemotherapy", "breast", "tumor", "neoplasm"]},

    # Endocrinology
    {"question": "What lifestyle interventions are most effective for type 2 diabetes prevention?",
     "domain": "endocrinology",
     "domain_keywords": ["diabetes", "glucose", "insulin", "glycemic", "metabolic"]},

    # Neurology
    {"question": "What are the current pharmacological treatments for Alzheimer's disease?",
     "domain": "neurology",
     "domain_keywords": ["alzheimer", "dementia", "cognitive", "neurological", "brain"]},

    # Psychiatry
    {"question": "How effective are SSRIs compared to therapy for treating depression?",
     "domain": "psychiatry",
     "domain_keywords": ["depression", "antidepressant", "ssri", "mental", "psychiatric"]},

    # Infectious diseases
    {"question": "What antibiotics are recommended for community-acquired pneumonia?",
     "domain": "infectious diseases",
     "domain_keywords": ["antibiotic", "pneumonia", "infection", "antimicrobial", "respiratory"]},

    # No domain filter (general)
    {"question": "What is the evidence for intermittent fasting in weight loss?",
     "domain": "",
     "domain_keywords": ["fasting", "weight", "obesity", "caloric", "diet"]},

    {"question": "What are the long-term effects of sleep deprivation on health?",
     "domain": "",
     "domain_keywords": ["sleep", "insomnia", "circadian", "fatigue", "cognitive"]},
]


# ──────────────────────────────────────────────
# GEMINI JUDGE
# ──────────────────────────────────────────────
def setup_gemini():
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client


def evaluate_relevance(model, question: str, title: str, abstract: str) -> dict:
    """
    Ask Gemini to score how relevant an article is to the question.
    Returns {"score": int 1-5, "reason": str}
    """
    prompt = f"""You are a medical research expert evaluating whether a PubMed article is relevant to a clinical question.

Clinical question: {question}

Article title: {title}

Article abstract: {abstract[:1200]}

Rate the relevance of this article to the clinical question on a scale of 1 to 5:
1 = Not relevant at all
2 = Slightly relevant (touches the topic but doesn't address the question)
3 = Moderately relevant (related topic but indirect answer)
4 = Highly relevant (directly addresses the question)
5 = Perfectly relevant (exactly what was asked, strong evidence)

Respond ONLY with valid JSON in this exact format:
{{"score": <integer 1-5>, "reason": "<one sentence explaining your score>"}}"""

    try:
        response = model.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text.strip()
        # Extract JSON even if surrounded by markdown fences
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = int(data.get("score", 3))
            score = max(1, min(5, score))  # clamp to [1,5]
            return {"score": score, "reason": data.get("reason", "")}
    except Exception as e:
        print(f"    [Gemini error] {e}")

    return {"score": 3, "reason": "Could not parse Gemini response."}


def check_domain_accuracy(abstract: str, title: str, keywords: list) -> bool:
    """
    Simple keyword check: does the article text contain at least one domain keyword?
    """
    text = (title + " " + abstract).lower()
    return any(kw.lower() in text for kw in keywords)


# ──────────────────────────────────────────────
# MAIN EVAL LOOP
# ──────────────────────────────────────────────
def run_evaluation():
    print("=" * 60)
    print("  RAGMed Evaluation  —  LLM-as-a-Judge (Gemini)")
    print("=" * 60)
    print(f"Model: {GEMINI_MODEL}")
    print(f"Queries: {len(TEST_QUERIES)}  |  Top-K per query: {TOP_K}")
    print()

    print("Loading embedding model (this may take a minute)...")
    reranker = SemanticReranker()
    print("Embedding model ready.\n")

    gemini = setup_gemini()

    all_results = []
    total_scores = []
    domain_correct = 0
    domain_total   = 0

    for i, tq in enumerate(TEST_QUERIES, 1):
        question   = tq["question"]
        domain     = tq["domain"]
        keywords   = tq["domain_keywords"]
        domain_label = domain if domain else "all PubMed"

        print(f"[{i}/{len(TEST_QUERIES)}] {question}")
        print(f"  Domain: {domain_label}")

        # --- Retrieve from PubMed ---
        try:
            rag_output = answer_with_pubmed_rag_v2(
                user_question=question,
                domain=domain,
                pubmed_retmax=PUBMED_RETMAX,
                final_k=TOP_K,
                reranker=reranker,
            )
        except Exception as e:
            print(f"  [RAG error] {e}")
            continue

        results = rag_output.get("results", [])
        n_candidates = rag_output.get("n_candidates", 0)
        print(f"  Candidates: {n_candidates}  |  Evaluating top {len(results)}")

        if not results:
            print("  No results returned.\n")
            all_results.append({
                "question": question,
                "domain": domain_label,
                "n_candidates": 0,
                "articles": [],
                "mean_relevance": None,
            })
            continue

        # --- Gemini scoring ---
        evaluated_articles = []
        query_scores = []

        for j, article in enumerate(results, 1):
            title    = article.get("title", "")
            abstract = article.get("abstract", "")

            # LLM relevance score
            evaluation = evaluate_relevance(gemini, question, title, abstract)
            score  = evaluation["score"]
            reason = evaluation["reason"]

            # Domain keyword accuracy
            if keywords:
                domain_hit = check_domain_accuracy(abstract, title, keywords)
                domain_correct += int(domain_hit)
                domain_total   += 1
            else:
                domain_hit = None

            query_scores.append(score)
            total_scores.append(score)

            print(f"    Article {j}: score={score}/5 | {title[:70]}...")
            print(f"             Reason: {reason}")

            evaluated_articles.append({
                "rank": j,
                "pmid": article.get("pmid", ""),
                "title": title,
                "journal": article.get("journal", ""),
                "year": article.get("year", ""),
                "pubmed_url": article.get("pubmed_url", ""),
                "semantic_score": article.get("semantic_score", 0),
                "recency_bonus": article.get("recency_bonus", 0),
                "combined_score": article.get("combined_score", 0),
                "gemini_relevance_score": score,
                "gemini_reason": reason,
                "domain_keyword_match": domain_hit,
            })

            time.sleep(1)  # rate limit buffer

        mean_q = round(sum(query_scores) / len(query_scores), 2) if query_scores else None
        print(f"  → Mean relevance: {mean_q}/5\n")

        all_results.append({
            "question": question,
            "domain": domain_label,
            "pubmed_query": rag_output.get("query_pubmed", ""),
            "n_candidates": n_candidates,
            "articles": evaluated_articles,
            "mean_relevance": mean_q,
        })

    # ── Summary ──
    overall_mean = round(sum(total_scores) / len(total_scores), 2) if total_scores else 0
    pct_relevant      = round(100 * sum(1 for s in total_scores if s >= 3) / len(total_scores), 1) if total_scores else 0
    pct_highly_relevant = round(100 * sum(1 for s in total_scores if s >= 4) / len(total_scores), 1) if total_scores else 0
    domain_accuracy   = round(100 * domain_correct / domain_total, 1) if domain_total else None

    summary = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_judge": GEMINI_MODEL,
        "n_queries": len(TEST_QUERIES),
        "top_k_per_query": TOP_K,
        "total_articles_evaluated": len(total_scores),
        "overall_mean_relevance": overall_mean,
        "pct_relevant_3_plus": pct_relevant,
        "pct_highly_relevant_4_plus": pct_highly_relevant,
        "domain_accuracy_pct": domain_accuracy,
        "score_distribution": {
            "5": sum(1 for s in total_scores if s == 5),
            "4": sum(1 for s in total_scores if s == 4),
            "3": sum(1 for s in total_scores if s == 3),
            "2": sum(1 for s in total_scores if s == 2),
            "1": sum(1 for s in total_scores if s == 1),
        },
        "per_query_results": all_results,
    }

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total articles evaluated : {len(total_scores)}")
    print(f"  Overall mean relevance   : {overall_mean}/5")
    print(f"  Relevant (score >= 3)    : {pct_relevant}%")
    print(f"  Highly relevant (>= 4)   : {pct_highly_relevant}%")
    if domain_accuracy is not None:
        print(f"  Domain accuracy          : {domain_accuracy}%")
    print()

    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {out_path}")
    return summary


if __name__ == "__main__":
    run_evaluation()
