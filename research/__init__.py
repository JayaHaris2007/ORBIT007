"""Modular web research engine for S.P.I.D.E.Y.

The pipeline is split into single-purpose modules so each stage can be run and
tested on its own:

    search_engine      Tavily search -> normalized, numbered sources
    page_reader        full page text + metadata for one URL        (milestone 2)
    evidence_checker   dedupe, relevance filter, conflict flags     (milestone 2)
    research_planner   question -> focused queries in deep mode     (milestone 6)
    answer_generator   evidence-only prompt -> cited answer         (milestone 3)

Only ``router`` is imported by ``app.py``; nothing here imports ``app``.
"""
