"""Conservative retrieval-time checks and source attribution helpers.

These checks require usable prose and enough topical overlap for the requested
subject. Generated claims are semantically validated in app.generation; lexical
overlap here ranks citation candidates only.
"""
import re
from dataclasses import replace

from .schemas import LearningSource


_STOP = set("a an and are as at be because been being by can could did do does for from had has have he her hers him his how i in into is it its may might of on or our should so that the their them then there these they this those through to was we were what when where which who why will with would you your happened describe tell explain called none both above below either all".split())
_GENERIC = set("answer answers chapter concept concepts course definition definitions document example examples fact facts following important importance information introduction learning material materials page question questions section topic topics test text theory".split())
# A bounded set of causal and functional constructions used only by retrieval-time
# filtering for "why" questions. Generated-claim entailment is handled by Ollama.
_CAUSAL_EVIDENCE = re.compile(
    r"\b(?:because|since|therefore|thus|as a result|consequently|due to|owing to|"
    r"leads? to|led to|results? in|resulted in|causes?|caused|enables?|enabled|"
    r"allows?|allowed|supports?|supported|contributes? to|depends? on|dependent on|"
    r"relies? on|used to|required to|necessary for|made possible)\b"
    r"|\bprovides?\b.{0,80}\bfor\b"
    r"|\bprovid(?:ed|ing)\b.{0,80}\b(?:for|to|that)\b"
    r"|\b(?:allows?|enabled?)\b.{0,80}\bto\b"
    r"|\bdependent on\b.{0,120}\bto\b"
    # These transitive patterns require a complement, not just a keyword anywhere.
    r"|\bensur(?:e|es|ed|ing)\b\s+(?:that\s+)?[a-z][\w-]*(?:\s+[a-z][\w-]*){0,7}"
    r"|\bmade\s+(?:it\s+)?possible\b"
    r"|\bhelp(?:s|ed|ing)?\b(?:\s+[a-z][\w-]*){1,5}\s+(?:to\b|[a-z][\w-]*\b)",
    re.I | re.S,
)
_NO_DEFINITION = re.compile(r"\b(?:does not|do not|doesn't|don't|not provide|without)\b.{0,80}\b(?:definition|definitions|axiom|axioms)\b", re.I)
_BOILERPLATE = re.compile(r"\b(?:intentionally explicit|grounding test questions|this material is|this document is|learning material|facts below are the only facts|only facts the learning system)\b", re.I)


def _tokens(text: str) -> list[str]:
    # PDF extraction sometimes inserts two spaces inside a word at a line-wrap
    # boundary (for example ``fl  oodplain``). Rejoin those fragments before
    # claim matching; ordinary word boundaries remain untouched.
    normalized = re.sub(r"\b([a-z]{1,2})[ \t]{2,}([a-z]{3,})\b", r"\1\2", text.casefold())
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", normalized)


def _normal(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith(("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    for suffix in ("ing", "ed"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def salient_tokens(text: str) -> set[str]:
    return {_normal(token) for token in _tokens(text) if token not in _STOP and token not in _GENERIC and len(token) > 2}


def _sentences(text: str) -> list[str]:
    # PDF extractors commonly put a newline at every printed line, including in the
    # middle of a sentence. Join those wrapped lines before evidence checks; keep
    # blank lines as paragraph boundaries and punctuation as sentence boundaries.
    text = re.sub(r"(?<![.!?])\n(?!\n)", " ", text)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if part.strip()]


def substantive_text(text: str) -> str:
    """Keep declarative prose; discard headings, question lists and boilerplate."""
    kept = []
    # Headings and question prompts can score in retrieval without stating a usable fact.
    for sentence in _sentences(text):
        if "?" in sentence or _BOILERPLATE.search(sentence):
            continue
        words = _tokens(sentence)
        if len(words) < 4 or len(salient_tokens(sentence)) < 3:
            continue
        if not re.search(r"[.!;:]", sentence) and not re.search(r"\b(is|are|was|were|has|have|had|includes|included|contains|called|began|became|established|abolished|created|represented|met|stormed|signed|declared|voted|formed|led|caused|provided|means|defined|requires|satisfies|extracts|extract)\b", sentence, re.I):
            continue
        kept.append(sentence)
    return " ".join(kept)


def task_evidence(query: str, chunks: list, task: str) -> list:
    """Filter retrieved chunks by actual declarative content and requested scope."""
    # Compare content words so generic query phrasing does not count as topical evidence.
    query_groups = [salient_tokens(part) for part in query.splitlines() if salient_tokens(part)]
    separate_exam_topics = task == "exam" and len(query_groups) > 1
    query_terms = set().union(*query_groups) if query_groups else salient_tokens(query)
    definition_request = bool(re.search(r"\b(define|definition|what is|what are|meaning|axiom)\b", query, re.I))
    causal_request = bool(re.search(r"\b(why|reason|cause|how come)\b", query, re.I))
    selected = []
    for chunk in chunks:
        content = substantive_text(chunk.content)
        if not content:
            continue
        full_text = chunk.content
        if definition_request and _NO_DEFINITION.search(full_text):
            continue
        evidence_terms = salient_tokens(content)
        overlap = query_terms & evidence_terms
        if query_terms:
            # The requested subject must be present; a broad topic match is not enough.
            # Cross-Encoder ranking handles semantic matches; retain a topical lexical floor
            # while allowing paraphrases such as "river cycle" for annual inundation.
            if separate_exam_topics:
                topic_matches = any(
                    bool(group) and len(group & evidence_terms) / len(group) >= (0.4 if len(group) > 2 else 1.0)
                    for group in query_groups
                )
                if not topic_matches:
                    continue
            elif len(overlap) / len(query_terms) < (0.4 if len(query_terms) > 2 else 1.0):
                continue
        if causal_request:
            # Require the causal construction and enough of the requested subject
            # to occur in the same substantive sentence. This prevents an unrelated
            # causal phrase elsewhere in a long chunk from satisfying a "why" query.
            sentence_threshold = 0.4 if len(query_terms) > 2 else 1.0
            if not any(
                _CAUSAL_EVIDENCE.search(sentence)
                and len(query_terms & salient_tokens(sentence)) / len(query_terms) >= sentence_threshold
                for sentence in _sentences(content)
            ):
                continue
        selected.append(replace(chunk, content=content))
    return selected


def rank_source_chunks(claim: str, chunks: list) -> list:
    """Rank evidence chunks for citation display; this is not a support decision."""
    terms = salient_tokens(claim)
    if not terms:
        return []
    ranked = []
    for chunk in chunks:
        evidence = salient_tokens(chunk.content)
        score = len(terms & evidence) / len(terms)
        if score > 0:
            ranked.append((score, chunk))
    return [chunk for _, chunk in sorted(ranked, key=lambda row: row[0], reverse=True)]


def source_for_claim(claim: str, chunks: list, *, semantically_accepted: bool = False) -> LearningSource | None:
    """Select a citation candidate after validation; token overlap never accepts a claim."""
    matches = rank_source_chunks(claim, chunks)
    terms = salient_tokens(claim)
    if matches:
        chunk = matches[0]
    elif semantically_accepted and chunks:
        # A semantic decision may cover the evidence packet without shared words.
        # In that case cite an actual accepted chunk rather than rejecting support.
        chunk = chunks[0]
    else:
        return None
    sentences = _sentences(chunk.content)
    snippet = max(sentences, key=lambda sentence: len(terms & salient_tokens(sentence)), default=chunk.content)
    if semantically_accepted and len(snippet) > 180:
        # A semantically matched chunk can begin with a page heading. Prefer the
        # bounded text window with the strongest overlap with the accepted claim.
        starts = [match.start() for match in re.finditer(r"\S+", snippet)]
        windows = [snippet[start:start + 180].strip() for start in starts]
        snippet = max(windows, key=lambda window: len(terms & salient_tokens(window)), default=snippet)
    metadata = getattr(chunk, "metadata", {}) or {}
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return LearningSource(document_name=metadata.get("source_filename") or chunk.document_name,
                          page_number=metadata.get("page") or chunk.page_number,
                          chunk_id=chunk.chunk_id, snippet=snippet[:240], relevance_score=chunk.relevance_score)
