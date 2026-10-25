"""Characterization tests for token budget context building and keyword memory search in memory.py."""

import pytest
from memory import MAX_CONTEXT_TOKENS, MAX_RETRIEVED_MEMORIES, MemoryStore


class FakeTokenizer:
    """Mock tokenizer simulating Hugging Face chat templating without needing GPU/Transformers."""

    def __init__(self, tokens_per_word: int = 1, fixed_overhead: int = 4):
        self.tokens_per_word = tokens_per_word
        self.fixed_overhead = fixed_overhead
        self.fail_template = False

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        if self.fail_template:
            raise TypeError("Simulated chat template failure for fallback testing")
        # Approximate 1 token per word + fixed overhead per message
        total_tokens = sum(
            len(m["content"].split()) * self.tokens_per_word + self.fixed_overhead
            for m in messages
        )
        return [0] * total_tokens


def test_build_context_fits_all(temp_db):
    chat = temp_db.create_chat()
    temp_db.add_message(chat["id"], "user", "Hello there")
    temp_db.add_message(chat["id"], "assistant", "Hi, how can I help you today?")
    temp_db.add_message(chat["id"], "user", "What's the status of my tasks?")

    tokenizer = FakeTokenizer()
    context = temp_db.build_context(chat["id"], tokenizer, "current query", max_tokens=100)
    assert len(context) == 3
    assert context[0]["role"] == "user"
    assert context[0]["content"] == "Hello there"
    assert context[-1]["content"] == "What's the status of my tasks?"


def test_build_context_truncates_oldest_first(temp_db):
    chat = temp_db.create_chat()
    # Add 5 messages
    for i in range(1, 6):
        role = "user" if i % 2 != 0 else "assistant"
        temp_db.add_message(chat["id"], role, f"Message number {i} with ten extra words to pad length")

    tokenizer = FakeTokenizer(tokens_per_word=1, fixed_overhead=2)
    # Set max_tokens small enough that only the 2 most recent messages fit
    context = temp_db.build_context(chat["id"], tokenizer, "next query", max_tokens=35)

    # Should only keep the most recent messages (4 and 5)
    assert len(context) < 5
    assert "Message number 5" in context[-1]["content"]
    assert "Message number 1" not in [m["content"] for m in context]


def test_build_context_tokenizer_fallback(temp_db):
    chat = temp_db.create_chat()
    temp_db.add_message(chat["id"], "user", "Simple short message")
    temp_db.add_message(chat["id"], "assistant", "Another response")

    tokenizer = FakeTokenizer()
    tokenizer.fail_template = True  # Triggers TypeError fallback

    # Fallback uses character estimation: len(content) // 4 + 8
    context = temp_db.build_context(chat["id"], tokenizer, "query", max_tokens=50)
    assert len(context) == 2


def test_memory_keyword_search_ranking(temp_db):
    # Ensure semantic embeddings are disabled for this test to test keyword fallback
    temp_db.add_memory("The user works primarily with Python and FastAPI backend development.")
    temp_db.add_memory("The user's favorite hobby is landscape photography in mountains.")
    temp_db.add_memory("The user writes Python scripts for automation and daily habit tracking.")

    # Search query targeting Python and development
    results = temp_db.search_memories("python backend development", limit=5)
    assert len(results) >= 2
    # The first result should be the one with the highest term overlap (Python, backend, development)
    assert "FastAPI backend" in results[0]["content"]
    # Photography should not be returned as it has 0 overlapping words
    assert not any("photography" in r["content"] for r in results)


def test_memory_keyword_search_case_insensitive_and_empty(temp_db):
    temp_db.add_memory("Project name is S.P.I.D.E.Y.")

    # Case-insensitive term search matches "project"
    results = temp_db.search_memories("project")
    assert len(results) == 1
    assert "S.P.I.D.E.Y" in results[0]["content"]

    # Characterization of current tokenizer behavior:
    # re.findall(r"[a-z0-9]+") splits dotted acronym "S.P.I.D.E.Y" into ['s', 'p', 'i', 'd', 'e', 'y']
    # Consequently, searching for "spidey" yields no intersection with the split single-letter tokens.
    assert temp_db.search_memories("spidey") == []

    # Empty query returns empty list
    empty_results = temp_db.search_memories("   ")
    assert empty_results == []


def test_memory_unique_constraint(temp_db):
    temp_db.add_memory("Unique fact about user")
    with pytest.raises(ValueError, match="already exists"):
        temp_db.add_memory("Unique fact about user")
