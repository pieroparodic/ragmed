import os
import re
from datetime import datetime
from typing import List, Dict, Optional

from Bio import Entrez
from dotenv import load_dotenv

# Embeddings / semantic re-ranking
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 0) CONFIG
# =========================================================
load_dotenv()

Entrez.email = os.getenv("NCBI_EMAIL", "user@example.com")  # required by NCBI
Entrez.tool = "pubmed_rag_v2"

api_key = os.getenv("NCBI_API_KEY")
if api_key:
    Entrez.api_key = api_key

# Lightweight, widely-used model for semantic embeddings
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Domain filters: each value is a PubMed query clause (MeSH + free-text).
# An empty string means no domain restriction — search across all of PubMed.
DOMAIN_FILTERS: Dict[str, str] = {
    "": "",  # No filter — all of PubMed
    "cardiology": (
        '("Cardiovascular Diseases"[MeSH Terms] '
        'OR cardiology[Title/Abstract] '
        'OR heart[Title/Abstract] '
        'OR cardiac[Title/Abstract])'
    ),
    "oncology": (
        '("Neoplasms"[MeSH Terms] '
        'OR cancer[Title/Abstract] '
        'OR tumor[Title/Abstract] '
        'OR oncology[Title/Abstract])'
    ),
    "neurology": (
        '("Nervous System Diseases"[MeSH Terms] '
        'OR neurology[Title/Abstract] '
        'OR neurological[Title/Abstract] '
        'OR brain[Title/Abstract])'
    ),
    "infectious diseases": (
        '("Communicable Diseases"[MeSH Terms] '
        'OR infection[Title/Abstract] '
        'OR infectious[Title/Abstract] '
        'OR antimicrobial[Title/Abstract])'
    ),
    "endocrinology": (
        '("Endocrine System Diseases"[MeSH Terms] '
        'OR diabetes[Title/Abstract] '
        'OR endocrine[Title/Abstract] '
        'OR hormone[Title/Abstract])'
    ),
    "pulmonology": (
        '("Respiratory Tract Diseases"[MeSH Terms] '
        'OR pulmonary[Title/Abstract] '
        'OR respiratory[Title/Abstract] '
        'OR lung[Title/Abstract])'
    ),
    "gastroenterology": (
        '("Gastrointestinal Diseases"[MeSH Terms] '
        'OR gastroenterology[Title/Abstract] '
        'OR gastrointestinal[Title/Abstract] '
        'OR hepatic[Title/Abstract])'
    ),
    "nephrology": (
        '("Kidney Diseases"[MeSH Terms] '
        'OR nephrology[Title/Abstract] '
        'OR renal[Title/Abstract] '
        'OR kidney[Title/Abstract])'
    ),
    "psychiatry": (
        '("Mental Disorders"[MeSH Terms] '
        'OR psychiatry[Title/Abstract] '
        'OR depression[Title/Abstract] '
        'OR mental health[Title/Abstract])'
    ),
    "rheumatology": (
        '("Rheumatic Diseases"[MeSH Terms] '
        'OR rheumatology[Title/Abstract] '
        'OR autoimmune[Title/Abstract] '
        'OR arthritis[Title/Abstract])'
    ),
}


# =========================================================
# 1) HELPERS
# =========================================================
def clean_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def build_pubmed_query(user_question: str, domain: str = "") -> str:
    """
    Build the PubMed query by combining:
    - an optional domain filter (MeSH + Title/Abstract terms)
    - the user's free-text question

    Tip: write the question in English for best PubMed recall.
    """
    user_question = clean_text(user_question)
    domain_clause = DOMAIN_FILTERS.get(domain, "")
    question_clause = f"({user_question})"

    if domain_clause:
        return f"{domain_clause} AND {question_clause}"
    return question_clause


# =========================================================
# 2) PUBMED RETRIEVAL (ESearch + EFetch)
# =========================================================
def search_pubmed_pmids(query: str, retmax: int = 20) -> List[str]:
    """
    ESearch -> returns a list of PMIDs.
    """
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=retmax,
        sort="relevance",
    )
    result = Entrez.read(handle)
    handle.close()

    return result.get("IdList", [])


def fetch_pubmed_records(pmids: List[str]) -> Dict:
    """
    EFetch -> returns full XML records.
    """
    if not pmids:
        return {}

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        retmode="xml",
    )
    records = Entrez.read(handle)
    handle.close()

    return records


# =========================================================
# 3) ARTICLE PARSING (TITLE / ABSTRACT / LINK / META)
# =========================================================
def extract_abstract(pubmed_article_record: Dict) -> str:
    """
    Extracts the abstract, preserving structured sections
    (Background, Methods, Results, etc.) when present.
    """
    article = pubmed_article_record["MedlineCitation"]["Article"]
    abstract_obj = article.get("Abstract")

    if not abstract_obj or "AbstractText" not in abstract_obj:
        return ""

    parts = []
    for section in abstract_obj["AbstractText"]:
        section_text = clean_text(section)
        if not section_text:
            continue

        label = ""
        # Biopython sometimes delivers attributes inside a StringElement
        if hasattr(section, "attributes") and isinstance(section.attributes, dict):
            label = clean_text(section.attributes.get("Label", ""))

        if label:
            parts.append(f"{label}: {section_text}")
        else:
            parts.append(section_text)

    return clean_text(" ".join(parts))


def extract_year(pubmed_article_record: Dict) -> str:
    """
    Tries to extract the publication year from several possible paths.
    """
    article = pubmed_article_record["MedlineCitation"]["Article"]
    journal = article.get("Journal", {})
    issue = journal.get("JournalIssue", {})
    pub_date = issue.get("PubDate", {})

    # Most common case: Year field
    if "Year" in pub_date:
        return clean_text(pub_date["Year"])

    # Edge case: MedlineDate (e.g. "2021 Jan-Feb")
    if "MedlineDate" in pub_date:
        md = clean_text(pub_date["MedlineDate"])
        m = re.search(r"(19|20)\d{2}", md)
        if m:
            return m.group(0)

    return ""


def extract_journal(pubmed_article_record: Dict) -> str:
    article = pubmed_article_record["MedlineCitation"]["Article"]
    journal = article.get("Journal", {})
    return clean_text(journal.get("Title", ""))


def extract_doi(pubmed_article_record: Dict) -> str:
    """
    Tries to extract the DOI from PubmedData -> ArticleIdList.
    """
    pubmed_data = pubmed_article_record.get("PubmedData", {})
    article_id_list = pubmed_data.get("ArticleIdList", [])

    for aid in article_id_list:
        if hasattr(aid, "attributes") and aid.attributes.get("IdType") == "doi":
            return clean_text(aid)

    return ""


def parse_pubmed_articles(xml_records: Dict, max_items: Optional[int] = None) -> List[Dict]:
    """
    Converts XML records into a structured list of article dicts.
    If max_items is None, returns all parseable articles that have an abstract.
    """
    parsed = []
    pubmed_articles = xml_records.get("PubmedArticle", [])

    for rec in pubmed_articles:
        try:
            medline = rec["MedlineCitation"]
            article = medline["Article"]

            pmid = clean_text(medline["PMID"])
            title = clean_text(article.get("ArticleTitle", ""))
            abstract = extract_abstract(rec)

            # Only keep articles that have an abstract
            if not abstract:
                continue

            journal = extract_journal(rec)
            year = extract_year(rec)
            doi = extract_doi(rec)

            parsed.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "doi": doi,
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )

            if max_items is not None and len(parsed) >= max_items:
                break

        except Exception:
            # A malformed record should not crash the whole pipeline
            continue

    return parsed


# =========================================================
# 4) LOCAL SEMANTIC RE-RANKING (EMBEDDINGS + RECENCY BOOST)
# =========================================================
# Weight applied to the recency bonus when computing the final score.
# final_score = semantic_score * (1 - RECENCY_WEIGHT) + recency_bonus * RECENCY_WEIGHT
# Setting this to 0 disables the recency bonus entirely.
RECENCY_WEIGHT: float = 0.15   # 15 % of the final score comes from recency
RECENCY_WINDOW_YEARS: int = 5  # Articles within this window get a full bonus


class SemanticReranker:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _build_doc_text(article: Dict) -> str:
        """Text fed to the embedding model (title + abstract)."""
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        return clean_text(f"{title}. {abstract}")

    @staticmethod
    def _recency_bonus(year_str: str) -> float:
        """
        Returns a score in [0, 1] based on how recent the article is.

        - Articles published in the last RECENCY_WINDOW_YEARS years → 1.0
        - Articles published exactly RECENCY_WINDOW_YEARS years ago  → 0.0
        - Older articles                                              → 0.0
        - Unknown year                                                → 0.0 (no penalty,
          the semantic score drives the ranking)

        The decay is linear within the recency window so that a 1-year-old
        article scores higher than a 4-year-old article.
        """
        if not year_str:
            return 0.0
        try:
            pub_year = int(year_str)
        except ValueError:
            return 0.0

        current_year = datetime.now().year
        age = current_year - pub_year  # years since publication

        if age < 0:
            # Future date — treat as current
            return 1.0
        if age >= RECENCY_WINDOW_YEARS:
            return 0.0
        # Linear decay: age=0 → 1.0, age=RECENCY_WINDOW_YEARS-1 → ~0.2
        return 1.0 - (age / RECENCY_WINDOW_YEARS)

    def rerank(self, query: str, articles: List[Dict], top_k: int = 3) -> List[Dict]:
        if not articles:
            return []

        doc_texts = [self._build_doc_text(a) for a in articles]
        query_text = clean_text(query)

        # Compute embeddings
        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        doc_embs = self.model.encode(doc_texts, normalize_embeddings=True)

        # Cosine similarity
        sims = cosine_similarity(query_emb, doc_embs)[0]

        # Combine semantic score with recency bonus
        scored = []
        for article, sem_score in zip(articles, sims):
            item = dict(article)
            recency = self._recency_bonus(article.get("year", ""))
            combined = (
                float(sem_score) * (1.0 - RECENCY_WEIGHT)
                + recency * RECENCY_WEIGHT
            )
            item["semantic_score"] = float(sem_score)
            item["recency_bonus"] = round(recency, 4)
            item["combined_score"] = round(combined, 4)
            scored.append(item)

        # Sort by combined score (semantic relevance + recency)
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored[:top_k]


# =========================================================
# 5) FULL PIPELINE V2
# =========================================================
def answer_with_pubmed_rag_v2(
    user_question: str,
    domain: str = "",
    pubmed_retmax: int = 20,
    final_k: int = 3,
    reranker: Optional[SemanticReranker] = None,
) -> Dict:
    """
    End-to-end pipeline:
      1) Build PubMed query (optional domain filter + user question)
      2) ESearch  -> retrieve PMIDs
      3) EFetch   -> fetch full XML records
      4) Parse articles that have an abstract
      5) Semantic re-ranking with local embeddings
      6) Return top-K results

    Returns a dict with metadata + results, suitable for API or CLI use.
    """
    if reranker is None:
        reranker = SemanticReranker()

    pubmed_query = build_pubmed_query(user_question, domain=domain)

    pmids = search_pubmed_pmids(pubmed_query, retmax=pubmed_retmax)
    if not pmids:
        return {
            "query_user": user_question,
            "query_pubmed": pubmed_query,
            "n_candidates": 0,
            "results": [],
        }

    xml_records = fetch_pubmed_records(pmids)
    candidates = parse_pubmed_articles(xml_records, max_items=None)

    # Re-rank using the original user question for best semantic match
    reranked = reranker.rerank(user_question, candidates, top_k=final_k)

    return {
        "query_user": user_question,
        "query_pubmed": pubmed_query,
        "n_candidates": len(candidates),
        "results": reranked,
    }


# =========================================================
# 6) TEXT OUTPUT FORMATTER (useful for CLI / web)
# =========================================================
def format_results_as_text(response: Dict, max_abstract_chars: int = 1800) -> str:
    lines = []
    lines.append("=== PubMed RAG v2 (Retrieval + Semantic Re-ranking) ===")
    lines.append(f"Question: {response.get('query_user', '')}")
    lines.append(f"PubMed query: {response.get('query_pubmed', '')}")
    lines.append(f"Candidates with abstract: {response.get('n_candidates', 0)}")
    lines.append("")

    results = response.get("results", [])
    if not results:
        lines.append("No articles with abstract found for this query.")
        return "\n".join(lines)

    for i, r in enumerate(results, 1):
        lines.append(f"--- Article {i} ---")
        lines.append(f"Title: {r.get('title', '')}")
        if r.get("journal") or r.get("year"):
            lines.append(f"Journal/Year: {r.get('journal', '')} ({r.get('year', '')})")
        lines.append(f"PMID: {r.get('pmid', '')}")
        lines.append(f"Link: {r.get('pubmed_url', '')}")
        if r.get("doi"):
            lines.append(f"DOI: {r.get('doi')}")
        lines.append(f"Semantic score: {r.get('semantic_score', 0):.4f}")
        lines.append(f"Recency bonus : {r.get('recency_bonus', 0):.4f}  (last {RECENCY_WINDOW_YEARS} yrs, weight {int(RECENCY_WEIGHT*100)}%)")
        lines.append(f"Combined score: {r.get('combined_score', r.get('semantic_score', 0)):.4f}")

        abstract = r.get("abstract", "")
        if len(abstract) > max_abstract_chars:
            abstract = abstract[:max_abstract_chars].rstrip() + "..."
        lines.append(f"Abstract: {abstract}")
        lines.append("")

    return "\n".join(lines)
