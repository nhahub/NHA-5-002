"""Assessment Agent (Member 2 of 5).
Extracted, upgraded, and productionized from Abdelrahman's Quiz Agent notebooks.
Features:
- Course & Lecture specific examination blueprint
- Comprehensive question pools (10 to 15+ questions covering definitions, proofs, state tables, edge cases)
- Dynamic generation from uploaded PDFs / text files
- Instant answer verification with pedagogical explanations
- Diagnostic error classification & weak concepts isolation feeding the Adaptive Personal Model
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from pypdf import PdfReader

from backend.app.config import settings

logger = logging.getLogger("adaptive_coach.assessment")

# Comprehensive question banks by course and lecture
COMPREHENSIVE_COURSE_BANKS = {
    "Automata & Formal Languages": {
        "Lecture 1: Introduction to Automata & DFA": [
            {
                "id": "aut-101",
                "topic": "Automata Basics",
                "question": "What is the formal definition of an Alphabet (Σ) in formal language theory?",
                "options": {
                    "A": "A finite, non-empty set of symbols",
                    "B": "An infinite set of strings of arbitrary length",
                    "C": "The set of all possible state transitions in an automaton",
                    "D": "A grammar containing both terminal and non-terminal symbols"
                },
                "correct_answer": "A",
                "explanation": "By definition, an alphabet Σ is a finite, non-empty set of atomic symbols or characters.",
                "concept": "Alphabet Definition",
                "cognitive_level": "Factual"
            },
            {
                "id": "aut-102",
                "topic": "DFA 5-Tuple",
                "question": "Which component of the 5-tuple M = (Q, Σ, δ, q0, F) represents the transition function of a DFA?",
                "options": {
                    "A": "Q: The set of final accept states",
                    "B": "δ: A mapping from Q × Σ → Q",
                    "C": "δ: A mapping from Q × (Σ ∪ {ε}) → 2^Q",
                    "D": "q0: The set of initial configurations"
                },
                "correct_answer": "B",
                "explanation": "In a Deterministic Finite Automaton (DFA), δ is a total function mapping Q × Σ → Q (exactly one next state for each state and symbol).",
                "concept": "DFA Formal Definition",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-103",
                "topic": "DFA Determinism",
                "question": "Why is a DFA called 'Deterministic'?",
                "options": {
                    "A": "Because it always halts in polynomial time",
                    "B": "Because for every state and input symbol, there is uniquely ONE designated next state",
                    "C": "Because it can guess non-deterministically which path leads to an accept state",
                    "D": "Because all of its states must be accepting states"
                },
                "correct_answer": "B",
                "explanation": "Determinism means that for any state q and input symbol a, the transition δ(q, a) yields exactly one specific state, with no ambiguity or guessing.",
                "concept": "Determinism Principle",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-104",
                "topic": "Language Recognition",
                "question": "When is an input string w = a1 a2 ... an accepted by a DFA M?",
                "options": {
                    "A": "If the automaton processes at least one symbol before halting",
                    "B": "If the sequence of transitions starting from q0 leads to a state r ∈ F upon consuming the entire string",
                    "C": "If the string contains an equal number of 0s and 1s",
                    "D": "If any visited intermediate state belongs to F"
                },
                "correct_answer": "B",
                "explanation": "Acceptance requires starting at initial state q0 and ending in an accept state r ∈ F precisely after the last input symbol is consumed.",
                "concept": "DFA Acceptance Criteria",
                "cognitive_level": "Application"
            },
            {
                "id": "aut-105",
                "topic": "DFA State Minimization",
                "question": "In a DFA with N states, what is the time complexity of running string recognition on an input of length L?",
                "options": {
                    "A": "O(N * L)",
                    "B": "O(2^N)",
                    "C": "O(L)",
                    "D": "O(L^2)"
                },
                "correct_answer": "C",
                "explanation": "Because each symbol transition is a single O(1) table lookup, processing a string of length L takes exactly O(L) linear time, regardless of N.",
                "concept": "DFA Recognition Complexity",
                "cognitive_level": "Application"
            },
            {
                "id": "aut-106",
                "topic": "Empty String Properties",
                "question": "How does a DFA decide whether the empty string ε is in its recognized language L(M)?",
                "options": {
                    "A": "If and only if the initial start state q0 is also an accept state (q0 ∈ F)",
                    "B": "If there is an epsilon transition from q0 to another state",
                    "C": "A DFA can never accept the empty string",
                    "D": "If all states in Q are accept states"
                },
                "correct_answer": "A",
                "explanation": "Reading ε means consuming zero symbols. The automaton remains at the initial state q0; therefore, ε ∈ L(M) if and only if q0 ∈ F.",
                "concept": "Empty String Acceptance",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-107",
                "topic": "Regular Languages",
                "question": "A formal language L is defined as 'Regular' if and only if:",
                "options": {
                    "A": "It can be generated by a Context-Free Grammar in Chomsky Normal Form",
                    "B": "There exists a Deterministic Finite Automaton that recognizes it",
                    "C": "It contains no infinite recursive patterns",
                    "D": "It can be parsed using a pushdown stack"
                },
                "correct_answer": "B",
                "explanation": "By Kleene's theorem and the definition of regularity, a language is regular if and only if it is recognized by a finite automaton (DFA or NFA).",
                "concept": "Regular Language Definition",
                "cognitive_level": "Factual"
            },
            {
                "id": "aut-108",
                "topic": "Dead / Trap States",
                "question": "What is the purpose of a 'Dead State' (or Trap State) in a DFA?",
                "options": {
                    "A": "A state that immediately resets the DFA back to the start state",
                    "B": "A non-accepting state whose transitions all loop back to itself, capturing invalid input prefixes",
                    "C": "An accepting state that can never be reached from q0",
                    "D": "A state used solely for handling epsilon transitions"
                },
                "correct_answer": "B",
                "explanation": "A dead state is a non-accepting state where all outbound transitions lead back to itself, ensuring that once an input violates language rules, it can never be accepted.",
                "concept": "Dead State Mechanics",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-109",
                "topic": "Complement of Regular Languages",
                "question": "To construct a DFA for the complement language L' = Σ* \\ L(M), what transformation is applied to DFA M = (Q, Σ, δ, q0, F)?",
                "options": {
                    "A": "Reverse all transition arrows δ",
                    "B": "Swap accept and non-accept states: F' = Q \\ F",
                    "C": "Add an epsilon transition from every state to q0",
                    "D": "Double the number of states Q"
                },
                "correct_answer": "B",
                "explanation": "Because a DFA is deterministic and fully specified for every symbol, inverting the set of final states (F' = Q \\ F) accepts all strings previously rejected and rejects all strings previously accepted.",
                "concept": "Complement Construction",
                "cognitive_level": "Application"
            },
            {
                "id": "aut-110",
                "topic": "State Minimization (Myhill-Nerode)",
                "question": "Two states p and q in a DFA are equivalent (indistinguishable) if and only if:",
                "options": {
                    "A": "They have identical numbers of incoming transitions",
                    "B": "For every string w ∈ Σ*, δ*(p, w) ∈ F if and only if δ*(q, w) ∈ F",
                    "C": "Both states have self-loops on the same symbol",
                    "D": "They can both reach the initial state q0"
                },
                "correct_answer": "B",
                "explanation": "Under the Myhill-Nerode relation, two states are indistinguishable if reading any arbitrary string w from either state always yields the same acceptance outcome.",
                "concept": "State Equivalence & Minimization",
                "cognitive_level": "Conceptual"
            }
        ],
        "Lecture 2: NFA & Epsilon Transitions": [
            {
                "id": "aut-201",
                "topic": "NFA Transition Function",
                "question": "How does the transition function δ of a Non-deterministic Finite Automaton (NFA) differ from a DFA?",
                "options": {
                    "A": "δ: Q × Σ → Q",
                    "B": "δ: Q × (Σ ∪ {ε}) → 2^Q (powersheet of states)",
                    "C": "δ: Q × Q → Σ",
                    "D": "δ maps states to probabilistic transition matrices"
                },
                "correct_answer": "B",
                "explanation": "An NFA's transition function maps a state and a symbol (or ε) to a set of states (a subset in 2^Q), allowing zero, one, or multiple target states.",
                "concept": "NFA Formal Definition",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-202",
                "topic": "Subset Construction Bound",
                "question": "For an NFA with k states, what is the maximum number of states in the equivalent DFA generated via subset construction?",
                "options": {
                    "A": "k^2",
                    "B": "2^k",
                    "C": "k!",
                    "D": "2k + 1"
                },
                "correct_answer": "B",
                "explanation": "Each state in the DFA corresponds to a subset of NFA states. A set of k states has 2^k possible subsets, establishing 2^k as the upper bound.",
                "concept": "Subset Construction Complexity",
                "cognitive_level": "Application"
            },
            {
                "id": "aut-203",
                "topic": "Epsilon Transitions",
                "question": "What is the primary operational characteristic of an epsilon (ε) transition in an NFA?",
                "options": {
                    "A": "It moves between states without consuming any symbol from the input string",
                    "B": "It erases the previous symbol read from the tape",
                    "C": "It halts the automaton and forces rejection",
                    "D": "It swaps the roles of start and final states"
                },
                "correct_answer": "A",
                "explanation": "An ε-transition allows the machine to transition spontaneously between states without reading a character from the input tape.",
                "concept": "Epsilon Transitions",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-204",
                "topic": "Epsilon Closure",
                "question": "The epsilon-closure ε-CLOSURE(q) of a state q is defined as:",
                "options": {
                    "A": "The set of states that can reach q using only normal alphabet symbols",
                    "B": "The set of all states reachable from q by following zero or more epsilon transitions",
                    "C": "The set of all accept states in the automaton",
                    "D": "Only the immediate neighboring states of q"
                },
                "correct_answer": "B",
                "explanation": "ε-CLOSURE(q) contains q itself plus every state reachable from q along any sequence of ε-transitions (transitive closure).",
                "concept": "Epsilon Closure Definition",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-205",
                "topic": "Equivalence of DFA and NFA",
                "question": "Does an NFA have greater computational expressive power than a DFA in terms of the classes of languages recognized?",
                "options": {
                    "A": "Yes, NFAs can recognize non-regular context-free languages",
                    "B": "No, every NFA can be converted to an equivalent DFA using subset construction, recognizing the exact same class of regular languages",
                    "C": "Yes, because NFAs can guess answers non-deterministically",
                    "D": "No, but DFAs can recognize languages that NFAs cannot"
                },
                "correct_answer": "B",
                "explanation": "NFAs and DFAs have identical computational power. Any NFA with ε-transitions can be converted to a DFA by subset construction.",
                "concept": "DFA-NFA Equivalence",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-206",
                "topic": "Pumping Lemma for Regular Languages",
                "question": "What is the primary purpose of the Pumping Lemma for Regular Languages?",
                "options": {
                    "A": "To prove that a given language is regular",
                    "B": "To minimize the number of states in an NFA",
                    "C": "To prove by contradiction that a given language is NOT regular",
                    "D": "To calculate the shortest path through a state diagram"
                },
                "correct_answer": "C",
                "explanation": "The Pumping Lemma is a necessary condition satisfied by all regular languages; it is used via proof by contradiction to establish that a language is not regular.",
                "concept": "Pumping Lemma Proofs",
                "cognitive_level": "Application"
            },
            {
                "id": "aut-207",
                "topic": "Pumping Lemma Conditions",
                "question": "In the Pumping Lemma, for any string s with |s| ≥ p, s can be split into s = xyz satisfying which conditions?",
                "options": {
                    "A": "|y| > 0, |xy| ≤ p, and xy^i z ∈ L for all i ≥ 0",
                    "B": "|x| = |y| = |z|, and x^i y^i z^i ∈ L",
                    "C": "|y| = 0, |xy| > p, and xyz ∈ L",
                    "D": "|xyz| ≤ p for all i ≥ 1"
                },
                "correct_answer": "A",
                "explanation": "The three necessary conditions are: 1. |y| > 0 (pumpable string is non-empty), 2. |xy| ≤ p (pumping happens within the first p characters), 3. xy^i z ∈ L for all i ≥ 0.",
                "concept": "Pumping Lemma Conditions",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-208",
                "topic": "Non-Regular Language Identification",
                "question": "Which of the following languages is provably NON-REGULAR?",
                "options": {
                    "A": "L = { w ∈ {0, 1}* | w contains an odd number of 0s }",
                    "B": "L = { 0^n 1^n | n ≥ 0 }",
                    "C": "L = { w | length of w is divisible by 5 }",
                    "D": "L = { w | w contains substring '101' }"
                },
                "correct_answer": "B",
                "explanation": "L = { 0^n 1^n } requires infinite counting memory to verify that the count of 0s equals the count of 1s, which exceeds the finite state capacity of a finite automaton.",
                "concept": "Non-Regular Language Examples",
                "cognitive_level": "Application"
            },
            {
                "id": "aut-209",
                "topic": "NFA Acceptance Definition",
                "question": "An NFA accepts an input string w if:",
                "options": {
                    "A": "Every possible execution branch ends in an accept state",
                    "B": "At least one valid computation path starting from q0 leads to a state in F upon reading w",
                    "C": "The majority of branches accept",
                    "D": "No branch enters a dead state"
                },
                "correct_answer": "B",
                "explanation": "Non-deterministic acceptance requires only existential satisfaction: if at least ONE valid branch ends in an accepting state, the string is accepted.",
                "concept": "NFA Acceptance Mechanics",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "aut-210",
                "topic": "Conversion Algorithm",
                "question": "When converting an NFA with states {1, 2, 3} to a DFA, when does a subset state {1, 2} become an accept state in the DFA?",
                "options": {
                    "A": "If all states in the subset ({1} and {2}) are accept states in the NFA",
                    "B": "If at least ONE state in the subset belongs to the NFA's final states F",
                    "C": "Only if {1, 2} has transitions back to itself",
                    "D": "If the subset contains the start state q0"
                },
                "correct_answer": "B",
                "explanation": "In subset construction, a DFA state S ⊆ Q is accepting if S ∩ F ≠ ∅ (i.e. contains at least one accepting state of the original NFA).",
                "concept": "Subset Construction Final States",
                "cognitive_level": "Application"
            }
        ]
    },
    "Trend AI & Modern Deep Learning": {
        "Lecture 1: AI Trends & Foundation Models": [
            {
                "id": "ai-101",
                "topic": "Foundation Models",
                "question": "What is the defining characteristic of a Foundation Model (e.g. GPT-4, Llama 3)?",
                "options": {
                    "A": "Trained on small curated tabular datasets using rule engines",
                    "B": "Trained on vast multi-modal/text corpora at scale and adaptable to diverse downstream tasks",
                    "C": "Designed exclusively for edge microcontroller deployment",
                    "D": "Requires manual hand-crafted feature engineering for each domain"
                },
                "correct_answer": "B",
                "explanation": "Foundation models are trained on massive broad datasets at scale using self-supervision and can be fine-tuned or prompted for a vast array of downstream tasks.",
                "concept": "Foundation Model Definition",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "ai-102",
                "topic": "Transformer Attention",
                "question": "In the self-attention formula Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V, what is the role of scaling by sqrt(d_k)?",
                "options": {
                    "A": "To invert the matrix dimensions to match the query size",
                    "B": "To prevent the dot products from growing excessively large for large dimensions, which would cause vanishing gradients in softmax",
                    "C": "To convert the attention weights into binary 0 or 1 values",
                    "D": "To enforce causal autoregressive masking"
                },
                "correct_answer": "B",
                "explanation": "For large d_k, dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Scaling by 1/sqrt(d_k) stabilizes training.",
                "concept": "Scaled Dot-Product Attention",
                "cognitive_level": "Application"
            },
            {
                "id": "ai-103",
                "topic": "Emergent Abilities",
                "question": "What are 'Emergent Abilities' in Large Language Models?",
                "options": {
                    "A": "Capabilities programmed explicitly into the loss function",
                    "B": "Abilities that are not present in smaller models but appear suddenly as model parameter scale and compute pass a critical threshold",
                    "C": "Hardware-level optimizations on TPU clusters",
                    "D": "Techniques for pruning neural network weights"
                },
                "correct_answer": "B",
                "explanation": "Emergent abilities are capabilities (e.g. in-context learning, multi-step chain-of-thought) that emerge unpredictably at massive compute/parameter scale.",
                "concept": "Emergence & Scaling Laws",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "ai-104",
                "topic": "Decoder-Only Architecture",
                "question": "Why do modern generative language models (like GPT and Llama) utilize Decoder-Only rather than Encoder-Decoder architectures?",
                "options": {
                    "A": "They cannot process bidirectional context during training",
                    "B": "Causal autoregressive decoding enables unified next-token prediction, pretraining efficiency, and zero-shot prompt completion",
                    "C": "Decoder-only models use 50% fewer attention heads",
                    "D": "They eliminate the need for embedding layers"
                },
                "correct_answer": "B",
                "explanation": "Decoder-only models unify generation and comprehension under a single causal next-token objective, maximizing parameter sharing and pretraining efficiency.",
                "concept": "Decoder-Only LLM Architecture",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "ai-105",
                "topic": "Hallucination in LLMs",
                "question": "What is the primary cause of 'Hallucination' in generative autoregressive language models?",
                "options": {
                    "A": "Hardware bit-flips in GPU memory",
                    "B": "The model generates statistically probable text continuation without grounded epistemic verification or fact-checking against external ground truth",
                    "C": "Using a temperature setting of exactly 0.0",
                    "D": "Exceeding the maximum sequence context length"
                },
                "correct_answer": "B",
                "explanation": "LLMs predict the most statistically likely next tokens based on pretraining data patterns, lacking intrinsic knowledge of truth or real-time ground truth verification.",
                "concept": "Hallucination Mechanics",
                "cognitive_level": "Conceptual"
            }
        ],
        "Lecture 2: RAG, Embeddings & Agents": [
            {
                "id": "ai-201",
                "topic": "RAG Core Pipeline",
                "question": "What are the three fundamental stages of a Retrieval-Augmented Generation (RAG) pipeline?",
                "options": {
                    "A": "Compile, Link, Execute",
                    "B": "Index & Ingest, Retrieve Relevant Chunks, Grounded Generation with LLM",
                    "C": "Tokenize, Train, Test",
                    "D": "Supervised Fine-Tuning, Reward Modeling, PPO"
                },
                "correct_answer": "B",
                "explanation": "RAG indexes documents into vectors/keywords, retrieves top-k relevant snippets given a query, and passes them as grounded context to an LLM generator.",
                "concept": "RAG Architecture",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "ai-202",
                "topic": "Hybrid Search",
                "question": "Why is Hybrid Retrieval (Dense Vector + Sparse BM25) superior to pure vector search in enterprise RAG systems?",
                "options": {
                    "A": "It eliminates the need for vector databases",
                    "B": "Dense embeddings capture semantic meaning while BM25 guarantees precision on exact keyword tokens (course codes, IDs, proper names)",
                    "C": "BM25 reduces latency by 90%",
                    "D": "It avoids chunking documents"
                },
                "correct_answer": "B",
                "explanation": "Pure vector search can fail on exact acronyms, IDs, or domain jargon. Blending dense semantic similarity with sparse BM25 provides optimal retrieval precision.",
                "concept": "Hybrid Search Mechanics",
                "cognitive_level": "Application"
            },
            {
                "id": "ai-203",
                "topic": "Re-Ranking Models",
                "question": "What is the role of a Cross-Encoder Re-Ranker in advanced RAG?",
                "options": {
                    "A": "To compress documents into smaller token sizes",
                    "B": "To jointly score the query and each candidate chunk with full cross-attention for high precision ranking before prompt construction",
                    "C": "To translate queries into SQL statements",
                    "D": "To generate the final student response directly"
                },
                "correct_answer": "B",
                "explanation": "Bi-encoders encode queries and documents separately for fast retrieval, but cross-encoders attend across both simultaneously to accurately re-rank top candidates.",
                "concept": "Cross-Encoder Re-Ranking",
                "cognitive_level": "Application"
            },
            {
                "id": "ai-204",
                "topic": "Multi-Agent Systems",
                "question": "What distinguishes an Agentic AI system from a traditional prompt-based LLM application?",
                "options": {
                    "A": "An Agent uses larger prompt templates",
                    "B": "An Agent exhibits autonomous perception, planning, tool usage, environment interaction, and stateful memory loops",
                    "C": "An Agent runs exclusively on local CPUs without GPU acceleration",
                    "D": "An Agent cannot produce structured JSON responses"
                },
                "correct_answer": "B",
                "explanation": "Agents possess agency: they perceive environments, decompose goals, call programmatic tools, update stateful memory, and loop autonomously until objectives are met.",
                "concept": "AI Agentic Architecture",
                "cognitive_level": "Conceptual"
            },
            {
                "id": "ai-205",
                "topic": "Behavioral Adaptation (The USP)",
                "question": "In an Adaptive AI Coach, what is the primary benefit of maintaining a continuous Personal Model of student behavior?",
                "options": {
                    "A": "It replaces the university curriculum with automated exams",
                    "B": "It anticipates cognitive fatigue, estimation bias, and procrastination patterns to adapt strategies preemptively rather than reactively rescheduling after failure",
                    "C": "It speeds up network download speeds for lecture PDFs",
                    "D": "It restricts study time to a fixed 15 minutes per day"
                },
                "correct_answer": "B",
                "explanation": "Conventional planners react only after tasks are missed. A behavioral personal model learns individual focus curves and patterns to intervene proactively before failure occurs.",
                "concept": "Adaptive Personal Modeling",
                "cognitive_level": "Conceptual"
            }
        ]
    }
}


class QuestionModel(BaseModel):
    id: str
    topic: str
    question: str
    options: Dict[str, str]
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str
    concept: str = ""
    cognitive_level: str = "Conceptual"


class AssessmentAgent:
    """Production Assessment Agent managing blueprints, generation, and diagnostic evaluation."""

    def __init__(self, material_dir: Optional[Path] = None):
        self.material_dir = material_dir or settings.material_dir

    def generate_quiz(
        self,
        course_name: str,
        lecture_title: Optional[str] = None,
        material_filename: Optional[str] = None,
        num_questions: int = 10,
        target_weakness: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive multi-question quiz for a specific course and lecture."""
        logger.info("Generating quiz for course: '%s', lecture: '%s', target count: %d", course_name, lecture_title, num_questions)

        # 1. Check curated course banks
        bank = self._get_curated_bank(course_name, lecture_title)
        if bank and len(bank) >= num_questions:
            questions = bank[:num_questions]
            return {
                "quiz_id": f"quiz-{uuid.uuid4().hex[:8]}",
                "course_name": course_name,
                "lecture_title": lecture_title or "Comprehensive Lecture Quiz",
                "total_questions": len(questions),
                "source": "curated_lecture_bank",
                "questions": questions
            }

        # 2. Try LLM generation if file or content exists
        if material_filename:
            llm_quiz = self._try_llm_generation(course_name, lecture_title, material_filename, num_questions, target_weakness)
            if llm_quiz:
                return llm_quiz

        # 3. Dynamic synthesis from PDF document text if available
        if material_filename:
            try:
                synthetic = self._synthesize_from_document(course_name, lecture_title, material_filename, num_questions)
                if synthetic and len(synthetic.get("questions", [])) >= 3:
                    return synthetic
            except Exception as e:
                logger.warning("Dynamic document quiz synthesis error: %s", e)

        # 4. Fallback to combining available questions
        fallback_pool = []
        if bank:
            fallback_pool.extend(bank)
        for cname, lectures in COMPREHENSIVE_COURSE_BANKS.items():
            if cname.lower() in course_name.lower() or course_name.lower() in cname.lower():
                for ltitle, qlist in lectures.items():
                    for q in qlist:
                        if q not in fallback_pool:
                            fallback_pool.append(q)

        if not fallback_pool and "automata" in course_name.lower():
            # Default Automata bank
            fallback_pool = COMPREHENSIVE_COURSE_BANKS["Automata & Formal Languages"]["Lecture 1: Introduction to Automata & DFA"]

        if not fallback_pool:
            return {
                "quiz_id": f"quiz-{uuid.uuid4().hex[:8]}",
                "course_name": course_name,
                "lecture_title": lecture_title or "Course Quiz",
                "total_questions": 0,
                "source": "no_course_material",
                "questions": []
            }

        questions = fallback_pool[:num_questions]
        return {
            "quiz_id": f"quiz-{uuid.uuid4().hex[:8]}",
            "course_name": course_name,
            "lecture_title": lecture_title or "Course Review Assessment",
            "total_questions": len(questions),
            "source": "adaptive_course_pool",
            "questions": questions
        }

    def _get_curated_bank(self, course_name: str, lecture_title: Optional[str]) -> List[Dict[str, Any]]:
        """Find matching questions in curated bank."""
        for cname, lectures in COMPREHENSIVE_COURSE_BANKS.items():
            if cname.lower() in course_name.lower() or course_name.lower() in cname.lower():
                if lecture_title:
                    for ltitle, qlist in lectures.items():
                        if ltitle.lower() in lecture_title.lower() or lecture_title.lower() in ltitle.lower():
                            return qlist
                # If lecture not specified or not matched, combine all lectures of this course
                combined = []
                for qlist in lectures.values():
                    combined.extend(qlist)
                return combined
        return []

    def _synthesize_from_document(
        self,
        course_name: str,
        lecture_title: Optional[str],
        filename: str,
        num_questions: int
    ) -> Dict[str, Any]:
        """Extract key paragraphs, definitions, and theorems from PDF to synthesize questions."""
        target_path = self.material_dir / filename
        if not target_path.exists():
            from backend.app.database import db
            target_path = db.uploads_dir / filename

        if not target_path.exists():
            # Try fuzzy search
            for f in self.material_dir.iterdir():
                if filename.lower() in f.name.lower():
                    target_path = f
                    break

        if not target_path.exists():
            return {}

        if target_path.suffix.lower() == ".pdf":
            reader = PdfReader(target_path)
            all_text = " ".join(page.extract_text() or "" for page in reader.pages)
        elif target_path.suffix.lower() == ".txt":
            all_text = target_path.read_text(encoding="utf-8", errors="ignore")
        elif target_path.suffix.lower() == ".docx":
            try:
                from docx import Document
                all_text = " ".join(p.text for p in Document(target_path).paragraphs)
            except ImportError:
                logger.warning("python-docx unavailable; cannot synthesize quiz from DOCX")
                return {}
        else:
            return {}
        sentences = [s.strip() for s in re.split(r"\. |\n\n", all_text) if 35 <= len(s.strip()) <= 180]

        questions = []
        for i, s in enumerate(sentences[:num_questions]):
            words = s.split()
            concept = " ".join(words[:4])
            questions.append({
                "id": f"dyn-{i+1}",
                "topic": course_name,
                "question": f"According to the lecture text, which of the following is verified: '{s[:120]}...'?",
                "options": {
                    "A": s[:85],
                    "B": f"The exact converse holds in all non-deterministic cases.",
                    "C": f"This property is strictly prohibited in {course_name}.",
                    "D": "None of the above statements are supported by the lecture."
                },
                "correct_answer": "A",
                "explanation": f"Directly stated in the course text: \"{s[:150]}...\".",
                "concept": concept,
                "cognitive_level": "Conceptual"
            })

        return {
            "quiz_id": f"quiz-{uuid.uuid4().hex[:8]}",
            "course_name": course_name,
            "lecture_title": lecture_title or filename,
            "total_questions": len(questions),
            "source": "document_extracted_quiz",
            "questions": questions
        }

    def _try_llm_generation(
        self,
        course_name: str,
        lecture_title: Optional[str],
        filename: str,
        num_questions: int,
        target_weakness: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Generate questions dynamically with Groq/OpenAI if configured."""
        api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None

        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=settings.groq_model,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            prompt = f"""
            You are an expert university professor creating an exhaustive diagnostic exam.
            Course: '{course_name}'
            Lecture: '{lecture_title or filename}'
            Generate {num_questions} high-yield multiple-choice questions (MCQs).
            Ensure coverage across definitions, algorithms, state proofs, and edge cases.
            Target weak concepts if provided: {target_weakness or 'Comprehensive lecture content'}.

            Return valid JSON with this exact schema:
            {{
                "questions": [
                    {{
                        "id": "q1",
                        "topic": "Subtopic name",
                        "question": "Question text...",
                        "options": {{
                            "A": "Option A text",
                            "B": "Option B text",
                            "C": "Option C text",
                            "D": "Option D text"
                        }},
                        "correct_answer": "A",
                        "explanation": "Clear pedagogical explanation of why A is right and others wrong.",
                        "concept": "Specific Concept Tested",
                        "cognitive_level": "Conceptual"
                    }}
                ]
            }}
            """
            res = llm.invoke(prompt)
            match = re.search(r"\{.*\}", res.content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                qs = data.get("questions", [])
                if len(qs) >= 3:
                    return {
                        "quiz_id": f"quiz-{uuid.uuid4().hex[:8]}",
                        "course_name": course_name,
                        "lecture_title": lecture_title or "LLM Generated Exam",
                        "total_questions": len(qs),
                        "source": "llm_generated_quiz",
                        "questions": qs
                    }
        except Exception as e:
            logger.warning("LLM quiz generation failed: %s", e)
        return None

    def evaluate_quiz(
        self,
        quiz_id: str,
        course_name: str,
        questions: List[Dict[str, Any]],
        user_answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Evaluate submitted answers, classify error types, isolate weak concepts."""
        score = 0
        total = len(questions)
        weak_topics = []
        error_types = {"Conceptual Gap": 0, "Application Error": 0, "Factual Slip": 0}
        detailed = []

        for q in questions:
            qid = str(q.get("id"))
            ans = user_answers.get(qid, "").strip().upper()
            correct = q.get("correct_answer", "A").strip().upper()
            is_correct = (ans == correct)

            if is_correct:
                score += 1
            else:
                concept = q.get("concept") or q.get("topic") or "General Theory"
                weak_topics.append(concept)
                level = q.get("cognitive_level", "Conceptual")
                if level == "Application":
                    error_types["Application Error"] += 1
                elif level == "Factual":
                    error_types["Factual Slip"] += 1
                else:
                    error_types["Conceptual Gap"] += 1

            detailed.append({
                "id": qid,
                "question": q.get("question"),
                "user_answer": ans,
                "correct_answer": correct,
                "is_correct": is_correct,
                "explanation": q.get("explanation"),
                "concept": q.get("concept")
            })

        pct = round((score / max(1, total)) * 100, 1)
        unique_weak = list(dict.fromkeys(weak_topics))

        summary = (
            f"You scored {score}/{total} ({pct}%). "
            + (f"Identified weak areas: {', '.join(unique_weak)}." if unique_weak else "Outstanding mastery! Zero conceptual mistakes detected.")
        )

        return {
            "quiz_id": quiz_id,
            "course_name": course_name,
            "score": score,
            "total": total,
            "score_pct": pct,
            "weak_topics": unique_weak,
            "error_types": {k: v for k, v in error_types.items() if v > 0},
            "detailed_answers": detailed,
            "feedback": summary
        }

assessment_agent = AssessmentAgent()
