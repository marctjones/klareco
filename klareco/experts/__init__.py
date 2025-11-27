"""
Expert modules for different response types.

This package contains specialized responders that handle different query types:
- ExtractiveResponder: Returns top retrieved sentence as answer
- ExtractiveSummarizer: Concatenates top-k sentences into a summary
"""
from klareco.experts.extractive import ExtractiveResponder, create_extractive_responder
from klareco.experts.summarizer import ExtractiveSummarizer, create_extractive_summarizer

__all__ = [
    "ExtractiveResponder",
    "create_extractive_responder",
    "ExtractiveSummarizer",
    "create_extractive_summarizer",
]
