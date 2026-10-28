"""Modular adaptive learning tracker for S.P.I.D.E.Y.

The pipeline is split into single-purpose modules so each stage can be run and
tested on its own:

    schema        learning tables + connection handling
    roadmaps      seed loader, prerequisites, next-topic queries
    classifier    task -> topic classification (rules, embeddings, LLM fallback)
    progress      completion/availability rules and activity logging
    nudges        weekly inactivity checks written to the notification bus
    router        HTTP surface mounted by ``app.py`` at /api/learning

Only ``router`` and ``schema`` are imported by ``app.py``; nothing here
imports ``app``.
"""
