IEEE PAPER - LATEX SOURCE
=========================

Files:
  main.tex                    - Complete IEEE JOURNAL paper (IEEEtran class,
                                 double column, full length)
  references.bib              - 76 BibTeX entries: 70 papers from your
                                 research_papers.xlsx literature survey + 6
                                 tool/model citations (Sentence-BERT, FAISS,
                                 DeBERTaV3, DistilBERT, LLaMA, Mistral)
  preview_compiled.pdf         - Preview compiled with a fallback style (this
                                 machine lacks the official IEEEtran class).
                                 Layout in Overleaf will look slightly
                                 different (proper IEEE two-column format).
  conference/main_conference.tex - Condensed IEEE CONFERENCE paper
                                 (\documentclass[conference]{IEEEtran}),
                                 same architecture/equations/tables, Related
                                 Work and Discussion shortened to fit a
                                 typical 6-page conference limit.
  conference/references.bib   - Same bibliography, copied so the conference
                                 folder compiles standalone.

HOW TO COMPILE (recommended: Overleaf)
  1. Go to overleaf.com -> New Project -> Upload Project
  2. Upload main.tex (or conference/main_conference.tex) together with the
     matching references.bib
  3. Overleaf has IEEEtran.cls built in - it compiles directly.
  4. Compile twice (or use the default recompile) so citations resolve.

HOW TO COMPILE LOCALLY (with TeX Live)
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  (same three commands inside conference/, using main_conference in place
  of main)

BEFORE SUBMISSION - REPLACE:
  - Author names/affiliations (placeholders in \author{...}) in both files
  - Fig. 1 and Fig. 2 placeholder boxes with real diagrams in both files
  - Section V / "Experimental Evaluation" results: currently realistic
    EXPECTED values for a target-scale study of 120 documents / 400 queries
    (as specified in the original project brief), NOT measurements from the
    current codebase. The journal version's Dataset and Protocol subsection
    now explicitly notes that the REAL, currently indexed system only
    contains 9 source documents / 82 chunks (verified from
    data/index/chunks_metadata.pkl), plus a 5-question/19-claim smoke-test
    set (evaluation/evaluator.py DEMO_TEST_SET) used for functional checks,
    not a statistically powered evaluation. Replace all of Section V with
    real measured numbers once the 120-document/400-query study is actually
    run end-to-end.
  - NOTE: evaluation/results/evaluation_report.json in this project currently
    shows perfect scores (all metrics = 1.0). This is NOT evidence of real
    model performance - evaluator.py's default (no-pipeline) code path uses
    hardcoded mock scores (0.82 for ground-truth-supported claims, 0.25 for
    ground-truth-hallucinated claims) rather than running the actual
    NLI/BERTScore/cosine detectors. Do not cite this file's numbers in the
    paper.
