"""Utility functions for Quorum proxy."""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def strip_thinking_tags(
    content: str, tags: List[str], hide_intermediate: bool = True
) -> str:
    """
    Strip reasoning tags from content based on configuration.

    Args:
        content: The text content to process
        tags: List of tag names to look for (e.g. ["think", "reason"])
        hide_intermediate: Whether to remove the text inside these tags

    Returns:
        Processed content with specified tags removed, if hide_intermediate is True
    """
    if not hide_intermediate:
        return content

    tag_pattern = "|".join(tags)
    pattern = f"<({tag_pattern})>.*?</\\1>"
    return re.sub(pattern, "", content, flags=re.IGNORECASE | re.DOTALL).strip()


class ThinkingTagFilter:
    """
    Incrementally removes content wrapped inside thinking tags (and their nested occurrences),
    while buffering partial tag input. If hide_intermediate_think is enabled, text in these
    tags is withheld from the streaming output. Final aggregator can also remove them
    depending on hide_final_think.
    """

    def __init__(self, tags):
        self.allowed_tags = [tag.lower() for tag in tags]
        tag_pattern = "|".join(self.allowed_tags)
        self.open_regex = re.compile(f"<({tag_pattern})>", re.IGNORECASE)
        self.close_regex = re.compile(f"</({tag_pattern})>", re.IGNORECASE)
        self.buffer = ""
        self.thinking_depth = 0

    def feed(self, text: str) -> str:
        """
        Add new text to the filter and return "safe" text outside thinking tags.
        """
        self.buffer += text
        output = ""

        while True:
            if self.thinking_depth == 0:
                open_match = self.open_regex.search(self.buffer)
                if open_match:
                    output += self.buffer[: open_match.start()]
                    self.buffer = self.buffer[open_match.start() :]
                    m = self.open_regex.match(self.buffer)
                    if m:
                        self.thinking_depth = 1
                        self.buffer = self.buffer[m.end() :]
                        continue
                    else:
                        pos = self.buffer.rfind("<")
                        if pos != -1:
                            candidate = self.buffer[pos:]
                            for tag in self.allowed_tags:
                                full_tag = f"<{tag}>"
                                if full_tag.startswith(candidate.lower()):
                                    output += self.buffer[:pos]
                                    self.buffer = self.buffer[pos:]
                                    return output
                        output += self.buffer
                        self.buffer = ""
                        break
                else:
                    pos = self.buffer.rfind("<")
                    if pos != -1:
                        candidate = self.buffer[pos:]
                        valid_partial = False
                        for tag in self.allowed_tags:
                            full_tag = f"<{tag}>"
                            if full_tag.startswith(candidate.lower()):
                                valid_partial = True
                                break
                        if valid_partial:
                            output += self.buffer[:pos]
                            self.buffer = self.buffer[pos:]
                            break
                    output += self.buffer
                    self.buffer = ""
                    break

            else:
                next_open = self.open_regex.search(self.buffer)
                next_close = self.close_regex.search(self.buffer)
                if not next_close and not next_open:
                    break
                if next_close and (
                    not next_open or next_close.start() < next_open.start()
                ):
                    self.buffer = self.buffer[next_close.end() :]
                    self.thinking_depth -= 1
                    if self.thinking_depth < 0:
                        self.thinking_depth = 0
                    continue
                elif next_open:
                    self.buffer = self.buffer[next_open.end() :]
                    self.thinking_depth += 1
                    continue
                else:
                    break

        return output

    def flush(self) -> str:
        """
        Flush any remaining "safe" text. If still inside a thinking block,
        discard that partial content.
        """
        if self.thinking_depth > 0:
            self.buffer = ""
            return ""
        else:
            pos = self.buffer.rfind("<")
            if pos != -1:
                candidate = self.buffer[pos:]
                for tag in self.allowed_tags:
                    full_tag = f"<{tag}>"
                    if full_tag.startswith(candidate.lower()):
                        self.buffer = self.buffer[:pos]
                        break
            out = self.buffer
            self.buffer = ""
            return out
