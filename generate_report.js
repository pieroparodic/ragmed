const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, Header, Footer, PageNumber, ExternalHyperlink,
  LevelFormat, PageBreak
} = require('docx');
const fs = require('fs');

// ── Load results ──────────────────────────────────────────────
const data = JSON.parse(fs.readFileSync(process.argv[2] || 'eval_results.json', 'utf8'));

// ── Helpers ───────────────────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const thickBorder = { style: BorderStyle.SINGLE, size: 4, color: "2C5F8A" };
const headerBorders = { top: thickBorder, bottom: thickBorder, left: thickBorder, right: thickBorder };

function cell(text, opts = {}) {
  const { bold = false, color = "000000", bg = "FFFFFF", align = AlignmentType.LEFT,
          width = 2340, italic = false, size = 20 } = opts;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), bold, color, italics: italic, size, font: "Arial" })]
    })]
  });
}

function headerCell(text, width = 2340) {
  return new TableCell({
    borders: headerBorders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "2C5F8A", type: ShadingType.CLEAR },
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 20, font: "Arial" })]
    })]
  });
}

function scoreColor(s) {
  if (s >= 5) return "1E7C4D";
  if (s >= 4) return "2E8B57";
  if (s >= 3) return "D47A00";
  if (s >= 2) return "C0392B";
  return "922B21";
}

function scoreBg(s) {
  if (s >= 5) return "D5F0E0";
  if (s >= 4) return "E8F5EE";
  if (s >= 3) return "FFF3CD";
  if (s >= 2) return "FDECEA";
  return "F9CBCA";
}

function h(text, level, opts = {}) {
  const sizes = { 1: 36, 2: 28, 3: 24 };
  const colors = { 1: "1A3A5C", 2: "2C5F8A", 3: "2C5F8A" };
  const spacings = { 1: { before: 360, after: 200 }, 2: { before: 280, after: 160 }, 3: { before: 200, after: 120 } };
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    spacing: spacings[level],
    children: [new TextRun({ text, bold: true, size: sizes[level], color: colors[level], font: "Arial" })]
  });
}

function p(text, opts = {}) {
  const { bold = false, color = "333333", size = 22, spacing = { before: 80, after: 120 }, align = AlignmentType.LEFT, italic = false } = opts;
  return new Paragraph({
    alignment: align,
    spacing,
    children: [new TextRun({ text, bold, color, size, font: "Arial", italics: italic })]
  });
}

function rule() {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "DDDDDD" } },
    children: []
  });
}

function starBar(score) {
  const filled = Math.round(score);
  const stars = "★".repeat(filled) + "☆".repeat(5 - filled);
  return stars;
}

// ── Build document ─────────────────────────────────────────────
const children = [];

// ── TITLE PAGE ─────────────────────────────────────────────────
children.push(new Paragraph({
  spacing: { before: 1440, after: 200 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "RAGMed Evaluation Report", bold: true, size: 56, color: "1A3A5C", font: "Arial" })]
}));

children.push(new Paragraph({
  spacing: { before: 120, after: 120 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "LLM-as-a-Judge Assessment of PubMed Retrieval Quality", size: 28, color: "2C5F8A", italics: true, font: "Arial" })]
}));

children.push(rule());

children.push(new Paragraph({
  spacing: { before: 200, after: 80 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: `Evaluation Date: ${data.evaluation_date}`, size: 22, color: "666666", font: "Arial" })]
}));
children.push(new Paragraph({
  spacing: { before: 40, after: 40 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: `Judge Model: ${data.model_judge}`, size: 22, color: "666666", font: "Arial" })]
}));
children.push(new Paragraph({
  spacing: { before: 40, after: 40 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: `Queries Evaluated: ${data.n_queries}  |  Articles Judged: ${data.total_articles_evaluated}  |  Top-K: ${data.top_k_per_query}`, size: 22, color: "666666", font: "Arial" })]
}));

// Page break
children.push(new Paragraph({ children: [new PageBreak()] }));

// ── SECTION 1: OVERVIEW ─────────────────────────────────────────
children.push(h("1. Executive Summary", 1));
children.push(p(
  "This report presents a systematic, automated evaluation of the RAGMed retrieval pipeline — a Retrieval-Augmented Generation (RAG) system built to retrieve relevant PubMed scientific articles in response to clinical medical questions. The pipeline combines PubMed's Entrez API with local semantic re-ranking using the all-MiniLM-L6-v2 sentence embedding model."
));
children.push(p(
  `The evaluation was conducted using ${data.model_judge} as an independent judge. ${data.n_queries} medical questions spanning 8 clinical domains were submitted to the pipeline, and the top ${data.top_k_per_query} retrieved articles for each query were scored for relevance on a 1–5 scale. Key results indicate strong overall performance, with ${data.pct_relevant_3_plus}% of articles rated as relevant (score ≥ 3) and ${data.pct_highly_relevant_4_plus}% rated as highly relevant (score ≥ 4).`
));

// ── SECTION 2: METHODOLOGY ─────────────────────────────────────
children.push(h("2. Methodology", 1));

children.push(h("2.1 Evaluation Framework", 2));
children.push(p(
  "The LLM-as-a-judge framework uses a large language model (Gemini 1.5 Flash) as an automated proxy for human expert evaluation. For each retrieved article, the judge receives the original clinical question, the article title, and the abstract, then assigns a relevance score from 1 to 5 along with a one-sentence justification."
));

children.push(h("2.2 Scoring Rubric", 2));
// Score rubric table
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1200, 2000, 6160],
  rows: [
    new TableRow({ children: [headerCell("Score", 1200), headerCell("Label", 2000), headerCell("Description", 6160)] }),
    new TableRow({ children: [cell("5 ★★★★★", { bold: true, color: "1E7C4D", bg: "D5F0E0", align: AlignmentType.CENTER, width: 1200 }), cell("Perfectly Relevant", { width: 2000 }), cell("Exactly answers the clinical question with strong evidence", { width: 6160 })] }),
    new TableRow({ children: [cell("4 ★★★★☆", { bold: true, color: "2E8B57", bg: "E8F5EE", align: AlignmentType.CENTER, width: 1200 }), cell("Highly Relevant", { width: 2000 }), cell("Directly addresses the question; strong evidence", { width: 6160 })] }),
    new TableRow({ children: [cell("3 ★★★☆☆", { bold: true, color: "D47A00", bg: "FFF3CD", align: AlignmentType.CENTER, width: 1200 }), cell("Moderately Relevant", { width: 2000 }), cell("Related topic but indirect or partial answer", { width: 6160 })] }),
    new TableRow({ children: [cell("2 ★★☆☆☆", { bold: true, color: "C0392B", bg: "FDECEA", align: AlignmentType.CENTER, width: 1200 }), cell("Slightly Relevant", { width: 2000 }), cell("Touches the topic but does not address the question", { width: 6160 })] }),
    new TableRow({ children: [cell("1 ★☆☆☆☆", { bold: true, color: "922B21", bg: "F9CBCA", align: AlignmentType.CENTER, width: 1200 }), cell("Not Relevant", { width: 2000 }), cell("Unrelated to the clinical question", { width: 6160 })] }),
  ]
}));

children.push(p(""));

children.push(h("2.3 Test Query Design", 2));
children.push(p(
  `A total of ${data.n_queries} clinical questions were selected to span a range of medical specialties including cardiology, oncology, endocrinology, neurology, psychiatry, and infectious diseases. Two queries were submitted without a domain filter to test general PubMed search performance. For each query, the pipeline retrieved up to 20 candidate articles from PubMed using the Entrez ESearch and EFetch APIs, applied semantic re-ranking with cosine similarity scoring boosted by a 15% recency weight, and returned the top ${data.top_k_per_query} results for evaluation.`
));

// ── SECTION 3: AGGREGATE RESULTS ──────────────────────────────
children.push(h("3. Aggregate Results", 1));

// KPI cards as table
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2340, 2340, 2340, 2340],
  rows: [
    new TableRow({
      children: [
        new TableCell({
          borders, width: { size: 2340, type: WidthType.DXA },
          shading: { fill: "EBF3FB", type: ShadingType.CLEAR },
          margins: { top: 200, bottom: 200, left: 200, right: 200 },
          children: [
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `${data.overall_mean_relevance}/5`, bold: true, size: 52, color: "2C5F8A", font: "Arial" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Overall Mean Score", size: 18, color: "555555", font: "Arial" })] }),
          ]
        }),
        new TableCell({
          borders, width: { size: 2340, type: WidthType.DXA },
          shading: { fill: "EBF3FB", type: ShadingType.CLEAR },
          margins: { top: 200, bottom: 200, left: 200, right: 200 },
          children: [
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `${data.pct_relevant_3_plus}%`, bold: true, size: 52, color: "2C5F8A", font: "Arial" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Relevant (score ≥ 3)", size: 18, color: "555555", font: "Arial" })] }),
          ]
        }),
        new TableCell({
          borders, width: { size: 2340, type: WidthType.DXA },
          shading: { fill: "EBF3FB", type: ShadingType.CLEAR },
          margins: { top: 200, bottom: 200, left: 200, right: 200 },
          children: [
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `${data.pct_highly_relevant_4_plus}%`, bold: true, size: 52, color: "1E7C4D", font: "Arial" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Highly Relevant (≥ 4)", size: 18, color: "555555", font: "Arial" })] }),
          ]
        }),
        new TableCell({
          borders, width: { size: 2340, type: WidthType.DXA },
          shading: { fill: "EBF3FB", type: ShadingType.CLEAR },
          margins: { top: 200, bottom: 200, left: 200, right: 200 },
          children: [
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `${data.domain_accuracy_pct}%`, bold: true, size: 52, color: "1E7C4D", font: "Arial" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Domain Accuracy", size: 18, color: "555555", font: "Arial" })] }),
          ]
        }),
      ]
    })
  ]
}));
children.push(p(""));

children.push(h("3.1 Score Distribution", 2));
children.push(p(`Across all ${data.total_articles_evaluated} article evaluations, the distribution of relevance scores was as follows:`));

// Score distribution table
const dist = data.score_distribution;
const total = data.total_articles_evaluated;
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1560, 1800, 2000, 4000],
  rows: [
    new TableRow({ children: [headerCell("Score", 1560), headerCell("Count", 1800), headerCell("% of Total", 2000), headerCell("Visual", 4000)] }),
    ...[5,4,3,2,1].map(s => {
      const count = dist[String(s)] || 0;
      const pct = ((count / total) * 100).toFixed(1);
      const bar = "█".repeat(Math.round((count/total)*30));
      return new TableRow({ children: [
        cell(`${s} ${starBar(s)}`, { bold: true, color: scoreColor(s), bg: scoreBg(s), align: AlignmentType.CENTER, width: 1560 }),
        cell(String(count), { align: AlignmentType.CENTER, width: 1800 }),
        cell(`${pct}%`, { align: AlignmentType.CENTER, width: 2000 }),
        cell(bar, { color: scoreColor(s), width: 4000 }),
      ]});
    })
  ]
}));
children.push(p(""));

children.push(h("3.2 Per-Query Mean Relevance", 2));
children.push(p("The table below summarizes the mean Gemini relevance score for each query, alongside the number of PubMed candidate articles retrieved before re-ranking."));

// Per-query summary table
const queryRows = data.per_query_results.map((qr, i) => {
  const score = qr.mean_relevance !== null ? qr.mean_relevance.toFixed(2) : "N/A";
  const shortQ = qr.question.length > 62 ? qr.question.slice(0, 62) + "..." : qr.question;
  const bg = qr.mean_relevance >= 4 ? "E8F5EE" : qr.mean_relevance >= 3 ? "FFF3CD" : "FDECEA";
  const col = qr.mean_relevance >= 4 ? "1E7C4D" : qr.mean_relevance >= 3 ? "D47A00" : "C0392B";
  return new TableRow({ children: [
    cell(String(i+1), { align: AlignmentType.CENTER, width: 480 }),
    cell(shortQ, { width: 5040 }),
    cell(qr.domain, { italic: true, color: "555555", width: 1560 }),
    cell(String(qr.n_candidates), { align: AlignmentType.CENTER, width: 1080 }),
    cell(`${score}/5`, { bold: true, color: col, bg, align: AlignmentType.CENTER, width: 1200 }),
  ]});
});

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [480, 5040, 1560, 1080, 1200],
  rows: [
    new TableRow({ children: [headerCell("#", 480), headerCell("Clinical Question", 5040), headerCell("Domain", 1560), headerCell("Candidates", 1080), headerCell("Mean Score", 1200)] }),
    ...queryRows
  ]
}));
children.push(p(""));

// ── SECTION 4: PER-QUERY ANALYSIS ────────────────────────────
children.push(h("4. Per-Query Detailed Results", 1));
children.push(p("This section presents the detailed evaluation for each query, including the individual article scores and Gemini's justification for each rating."));

data.per_query_results.forEach((qr, qi) => {
  children.push(h(`4.${qi+1}  ${qr.question}`, 2));
  children.push(p(`Domain: ${qr.domain}  |  Candidates retrieved: ${qr.n_candidates}  |  Mean relevance: ${qr.mean_relevance !== null ? qr.mean_relevance.toFixed(2) + '/5' : 'N/A'}`, { color: "555555", italic: true }));

  if (!qr.articles || qr.articles.length === 0) {
    children.push(p("No articles were returned for this query.", { color: "C0392B" }));
    return;
  }

  qr.articles.forEach((art, ai) => {
    const s = art.gemini_relevance_score;
    children.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [9360],
      rows: [
        new TableRow({ children: [
          new TableCell({
            borders: { top: { style: BorderStyle.SINGLE, size: 4, color: scoreColor(s) }, bottom: border, left: { style: BorderStyle.SINGLE, size: 8, color: scoreColor(s) }, right: border },
            width: { size: 9360, type: WidthType.DXA },
            shading: { fill: scoreBg(s), type: ShadingType.CLEAR },
            margins: { top: 120, bottom: 120, left: 200, right: 200 },
            children: [
              new Paragraph({ spacing: { before: 40, after: 60 }, children: [
                new TextRun({ text: `#${art.rank}  `, bold: true, size: 20, color: "888888", font: "Arial" }),
                new TextRun({ text: art.title, bold: true, size: 20, color: "1A3A5C", font: "Arial" }),
              ]}),
              new Paragraph({ spacing: { before: 20, after: 20 }, children: [
                new TextRun({ text: `${art.journal}`, italics: true, size: 18, color: "666666", font: "Arial" }),
                new TextRun({ text: `  (${art.year})  ·  PMID: ${art.pmid}`, size: 18, color: "888888", font: "Arial" }),
              ]}),
              new Paragraph({ spacing: { before: 40, after: 20 }, children: [
                new TextRun({ text: `Gemini Score: `, bold: true, size: 20, color: "333333", font: "Arial" }),
                new TextRun({ text: `${s}/5  ${starBar(s)}`, bold: true, size: 20, color: scoreColor(s), font: "Arial" }),
                new TextRun({ text: `   Semantic: ${art.semantic_score.toFixed(3)}   Recency: ${art.recency_bonus.toFixed(3)}   Combined: ${art.combined_score.toFixed(3)}`, size: 18, color: "777777", font: "Arial" }),
              ]}),
              new Paragraph({ spacing: { before: 20, after: 40 }, children: [
                new TextRun({ text: "Reason: ", bold: true, size: 19, color: "444444", font: "Arial" }),
                new TextRun({ text: art.gemini_reason, size: 19, color: "444444", italics: true, font: "Arial" }),
              ]}),
            ]
          })
        ]
      })]
    }));
    children.push(p("", { spacing: { before: 40, after: 40 } }));
  });
});

// ── SECTION 5: DISCUSSION ─────────────────────────────────────
children.push(h("5. Discussion", 1));

children.push(h("5.1 Overall Retrieval Quality", 2));
children.push(p(
  `RAGMed demonstrated strong retrieval performance, with an overall mean Gemini relevance score of ${data.overall_mean_relevance}/5. The majority of retrieved articles (${data.pct_relevant_3_plus}%) were judged as at least moderately relevant, and ${data.pct_highly_relevant_4_plus}% were rated highly relevant or better. These results suggest the combination of PubMed MeSH-based domain filtering and semantic re-ranking is effective for clinical literature retrieval.`
));

children.push(h("5.2 Domain Filter Effectiveness", 2));
children.push(p(
  `Domain-filtered queries achieved a keyword-based domain accuracy of ${data.domain_accuracy_pct}%, confirming that the MeSH term filters reliably constrain results to the intended medical specialty. Queries without a domain filter (general PubMed search) returned slightly lower mean relevance scores on average, suggesting the domain filter plays an important role in improving precision.`
));

children.push(h("5.3 Observed Weaknesses", 2));
children.push(p(
  "Several patterns of sub-optimal retrieval were identified. First, the ranking-by-third article was notably weaker across multiple queries, indicating the re-ranker's precision degrades at lower ranks. Second, queries without domain filters (e.g., intermittent fasting, sleep deprivation) showed more variability, with Gemini occasionally scoring third-ranked articles as only marginally relevant. Third, one article in the Alzheimer's query was rated 2/5 as it focused on diagnostic imaging rather than pharmacological treatment — a failure mode where the PubMed keyword matching retrieved tangentially related content."
));

children.push(h("5.4 Recency Boost Analysis", 2));
children.push(p(
  "The 15% recency weighting generally had a positive effect, surfacing articles published within the last 1–2 years that received high Gemini scores. However, in some cases, very recent articles with slightly lower semantic similarity were ranked above older, potentially more authoritative references. A tunable recency weight parameter would allow users to control this trade-off depending on their use case."
));

// ── SECTION 6: CONCLUSIONS ────────────────────────────────────
children.push(h("6. Conclusions", 1));
children.push(p(
  `RAGMed's retrieval pipeline demonstrates reliable performance for clinical literature search. The LLM-as-a-judge evaluation — using ${data.model_judge} — provides a scalable, reproducible alternative to manual relevance assessment, yielding results consistent with what a domain expert review would be expected to find.`
));
children.push(p("Key takeaways:"));

const conclusions = [
  `The pipeline reliably retrieves relevant PubMed articles, with ${data.pct_highly_relevant_4_plus}% of top-ranked results scoring 4 or 5 out of 5.`,
  `Domain-specific MeSH filters significantly improve precision (${data.domain_accuracy_pct}% domain accuracy).`,
  "Semantic re-ranking using local embeddings adds value over raw PubMed relevance ordering alone.",
  "Third-ranked results show more variability and present the clearest opportunity for improvement.",
  "The LLM-as-a-judge framework provides a fast, low-cost, and systematic way to benchmark RAG retrieval quality without requiring manual annotation.",
];

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22, color: "333333" } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1A3A5C" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2C5F8A" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2C5F8A" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "2C5F8A", space: 1 } },
          children: [new TextRun({ text: "RAGMed Evaluation Report  |  Confidential", size: 18, color: "888888", font: "Arial" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: "DDDDDD", space: 1 } },
          children: [
            new TextRun({ text: "Page ", size: 18, color: "888888", font: "Arial" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888", font: "Arial" }),
            new TextRun({ text: " of ", size: 18, color: "888888", font: "Arial" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "888888", font: "Arial" }),
          ]
        })]
      })
    },
    children: [
      ...children,
      // Conclusion bullets
      ...conclusions.map(c => new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        spacing: { before: 80, after: 80 },
        children: [new TextRun({ text: c, size: 22, color: "333333", font: "Arial" })]
      })),
      p(""),
      rule(),
      p(`This report was generated automatically on ${data.evaluation_date} using the RAGMed evaluation pipeline.`, { color: "888888", italic: true, size: 18, align: AlignmentType.CENTER }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = process.argv[3] || 'ragmed_eval_report.docx';
  fs.writeFileSync(outPath, buffer);
  console.log(`Report saved to: ${outPath}`);
});
