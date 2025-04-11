import json
import pytest
from grappa import should
from quorum.oai_proxy import ThinkingTagFilter


def test_thinking_tag_filter_basic():
    """
    Basic functionality:
      - A single complete thinking block.
      - Multiple blocks in sequence.
    """
    filt = ThinkingTagFilter(["think", "reason", "reasoning", "thought"])
    text = "Hello <think>secret</think> World"
    result = filt.feed(text)
    result | should.equal("Hello  World")

    filt = ThinkingTagFilter(["think", "reason", "reasoning", "thought"])
    text2 = "A <think>block1</think> B <think>block2</think> C"
    result2 = filt.feed(text2)
    result2 | should.equal("A  B  C")


def test_thinking_tag_filter_split_tags():
    """
    Split tags across feeds:
      - An opening tag split across two feeds.
      - A closing tag split across feeds.
      - No output from within an unclosed thinking block.
    """
    filt = ThinkingTagFilter(["think"])
    # Feed a chunk that ends with a partial open tag.
    out1 = filt.feed("Hello <thi")
    out1 | should.equal("Hello ")
    # Now feed the remainder to complete the open tag and include content.
    out2 = filt.feed("nk>secret</th")
    out2 | should.equal("")
    # Finally, complete the closing tag and add safe text.
    out3 = filt.feed("ink> World")
    out3 | should.equal(" World")


def test_thinking_tag_filter_nested_tags():
    """
    Nested tags:
      - When a tag opens inside an already opened tag (same type), the entire block is treated as thinking content.
      - Nested tags of different types are likewise fully removed.
    """
    filt = ThinkingTagFilter(["think", "reason"])
    text = "A <think>first <think>inner</think> still in</think> D"
    result = filt.feed(text)
    result | should.equal("A  D")

    filt = ThinkingTagFilter(["think", "reason"])
    text2 = "X <think>hello <reason>ignore</reason> world</think> Y"
    result2 = filt.feed(text2)
    result2 | should.equal("X  Y")


def test_thinking_tag_filter_incomplete_tags():
    """
    Incomplete/malformed tags:
      - Unclosed tags: output only safe text and withhold thinking content.
      - Mismatched closing tags: if a closing tag doesn’t match an open tag, treat the block as unclosed.
    """
    # Unclosed tag scenario.
    filt = ThinkingTagFilter(["think"])
    result = filt.feed("Hello <think>this is not closed")
    result | should.equal("Hello ")
    flush_result = filt.flush()
    flush_result | should.equal("")

    # Mismatched closing tag: </nope> is not recognized for allowed tag "think".
    filt = ThinkingTagFilter(["think"])
    result = filt.feed("Test <think>secret</nope> End")
    result | should.equal("Test ")
    flush_result = filt.flush()
    flush_result | should.equal("")


def test_thinking_tag_filter_case_insensitive():
    """
    Case sensitivity:
      - Uppercase tags and mixed-case tags must be recognized.
    """
    filt = ThinkingTagFilter(["think"])
    text = "Hello <THINK>Secret</THINK> World"
    result = filt.feed(text)
    result | should.equal("Hello  World")

    filt = ThinkingTagFilter(["think"])
    text2 = "Hello <ThInK>Secret</tHiNk> World"
    result2 = filt.feed(text2)
    result2 | should.equal("Hello  World")


def test_thinking_tag_filter_flush():
    """
    The flush() method:
      - When no thinking block is open, flush() returns any leftover safe text.
      - When a tag remains unclosed, flush() returns an empty string.
      - The internal buffer is then cleared.
    """
    # No tags: flush returns nothing extra.
    filt = ThinkingTagFilter(["think"])
    text = "No tags here."
    result = filt.feed(text)
    result | should.equal("No tags here.")
    flush_result = filt.flush()
    flush_result | should.equal("")

    # Open (partial) tag: feed returns safe text; flush() discards the incomplete tag.
    filt = ThinkingTagFilter(["think"])
    result = filt.feed("Partial open <think")
    result | should.equal("Partial open ")
    flush_result = filt.flush()
    flush_result | should.equal("")


def test_thinking_tag_filter_streaming_simulation():
    """
    Simulate realistic streaming with multiple chunks:
      - Partial tag boundaries spanning chunks.
      - Verify that the safe text is output in order and no thinking content is leaked.
    """
    filt = ThinkingTagFilter(["think"])
    # Chunk 1: safe text plus a partial tag.
    out1 = filt.feed("Stream start <thin")
    out1 | should.equal("Stream start ")
    # Chunk 2: complete the open tag and part of the inner thinking text.
    out2 = filt.feed("k>secret mess")
    out2 | should.equal("")
    # Chunk 3: complete the closing tag and add safe text.
    out3 = filt.feed("age</think> and then safe")
    out3 | should.equal(" and then safe")


def test_thinking_tag_filter_multiple_tags():
    """
    Multiple tag types:
      - Ensure that if more than one tag type is configured (e.g. <think> and <reason>),
        both are filtered out.
      - Interleaved safe content is preserved.
    """
    filt = ThinkingTagFilter(["think", "reason"])
    text = "Hello <think>skip</think> world <reason>ignore</reason> done"
    result = filt.feed(text)
    result | should.equal("Hello  world  done")

    filt = ThinkingTagFilter(["think", "reason"])
    text2 = "Start <think>remove this</think> Middle <reason>remove that</reason> End"
    result2 = filt.feed(text2)
    result2 | should.equal("Start  Middle  End")


def test_thinking_tag_filter_with_newlines():
    """
    Newline handling:
      - Multi‐line content inside tags must be removed.
      - Tags split across lines are correctly detected.
      - Safe newlines in the output are preserved.
    """
    filt = ThinkingTagFilter(["think"])
    text = "Line1\n<think>should be removed\nstill removed</think>\nLine2"
    result = filt.feed(text)
    result | should.equal("Line1\n\nLine2")

    # Simulate tags split across lines.
    filt = ThinkingTagFilter(["think"])
    out1 = filt.feed("Hello <thin")
    out1 | should.equal("Hello ")
    out2 = filt.feed("k>\nsecret\n")
    out2 | should.equal("")
    out3 = filt.feed("content</think>\nWorld")
    out3 | should.equal("\nWorld")