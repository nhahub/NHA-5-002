"""Run a fixed regression benchmark against the local Ollama judge."""
from dataclasses import dataclass

from app.config import get_settings
from app.domain import GroundedContext, RetrievedChunk
from app.generation import LearningGenerator
from app.providers import OllamaProvider
from app.rag import InMemoryRetriever


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    topic: str
    question: str
    evidence: str
    claim: str
    expected_supported: bool


CASES = [
    BenchmarkCase("clearly_supported", "French Revolution", "When did it begin?",
                  "The French Revolution began in 1789.",
                  "The French Revolution began in 1789.", True),
    BenchmarkCase("clearly_supported", "French Revolution", "What happened at the Bastille?",
                  "Revolutionaries stormed the Bastille on 14 July 1789.",
                  "Revolutionaries stormed the Bastille on 14 July 1789.", True),
    BenchmarkCase("clearly_supported", "French Revolution", "What did the Declaration establish?",
                  "The Declaration of the Rights of Man established liberty, equality, and political rights.",
                  "The Declaration of the Rights of Man established liberty, equality, and political rights.", True),
    BenchmarkCase("clearly_supported", "Thermal conductivity", "How does copper transfer heat?",
                  "Copper conducts heat efficiently because free electrons transfer energy.",
                  "Copper conducts heat efficiently because free electrons transfer energy.", True),
    BenchmarkCase("clearly_supported", "Vector spaces", "What is a vector space?",
                  "A vector space is a set with addition and scalar multiplication that satisfies the vector space axioms.",
                  "A vector space is a set with addition and scalar multiplication that satisfies the vector space axioms.", True),

    BenchmarkCase("faithful_paraphrase", "Ancient Egyptian agriculture", "Why did the inundation matter?",
                  "Ancient Egypt depended on the annual inundation of the Nile River to support its agricultural population.",
                  "The yearly Nile flood sustained Egypt's farming population.", True),
    BenchmarkCase("faithful_paraphrase", "Ancient Egyptian agriculture", "What supported productivity?",
                  "The fertility of the land and general predictability of the inundation ensured very high productivity from a single annual crop.",
                  "Predictable flooding and fertile land supported high yields from one crop each year.", True),
    BenchmarkCase("faithful_paraphrase", "French Revolution", "Why is the Bastille event important?",
                  "The event became an important symbol of the Revolution.",
                  "The Bastille event came to represent the Revolution.", True),
    BenchmarkCase("faithful_paraphrase", "Thermal conductivity", "How is heat transferred in copper?",
                  "Copper conducts heat efficiently because free electrons transfer energy.",
                  "Free electrons carry energy through copper, helping it conduct heat efficiently.", True),
    BenchmarkCase("faithful_paraphrase", "French Revolution", "What principles were established?",
                  "The Declaration of the Rights of Man established liberty, equality, and political rights.",
                  "The Declaration laid down principles of liberty, equality, and political rights.", True),

    BenchmarkCase("clearly_unsupported", "Ancient Egyptian agriculture", "Why did the Nile matter?",
                  "Annual inundation of the Nile supported Egypt's agricultural population.",
                  "The Nile helped Egyptian armies conquer neighboring countries.", False),
    BenchmarkCase("clearly_unsupported", "French Revolution", "Who stormed the Bastille and when?",
                  "Revolutionaries stormed the Bastille on 14 July 1789.",
                  "Napoleon captured the Bastille in 1804.", False),
    BenchmarkCase("clearly_unsupported", "French Revolution", "What rights were in the Declaration?",
                  "The Declaration of the Rights of Man established liberty, equality, and political rights.",
                  "The Declaration granted women universal voting rights.", False),
    BenchmarkCase("clearly_unsupported", "Thermal conductivity", "What does copper do?",
                  "Copper conducts heat efficiently because free electrons transfer energy.",
                  "Copper cures infectious diseases.", False),
    BenchmarkCase("clearly_unsupported", "Vector spaces", "What conditions define a vector space?",
                  "A vector space is a set with addition and scalar multiplication that satisfies the vector space axioms.",
                  "Every vector space contains a matrix inverse for each element.", False),

    BenchmarkCase("plausible_but_unsupported", "Ancient Egyptian agriculture", "What did the flood leave?",
                  "The annual inundation supported Egypt's agricultural population.",
                  "The Nile deposited a thin layer of nutrient-rich silt on the fields.", False),
    BenchmarkCase("plausible_but_unsupported", "Ancient Egyptian agriculture", "How many crops were grown?",
                  "The predictability of the inundation ensured high productivity from a single annual crop.",
                  "Predictable Nile floods allowed Egyptian farmers to harvest two crops every year.", False),
    BenchmarkCase("plausible_but_unsupported", "Ancient Egyptian agriculture", "How were fields irrigated?",
                  "Annual inundation supported the agricultural population of Egypt.",
                  "Pharaohs built irrigation canals to store floodwater for dry months.", False),
    BenchmarkCase("plausible_but_unsupported", "Ancient Egyptian agriculture", "What did farmers export?",
                  "Agriculture centered on cereal crops, chiefly emmer wheat and barley.",
                  "Egyptian farmers exported wheat to Greece along the Nile trade route.", False),
    BenchmarkCase("plausible_but_unsupported", "Thermal conductivity", "How do electrons move?",
                  "Copper conducts heat efficiently because free electrons transfer energy.",
                  "Free electrons move from hot regions to cold regions inside copper.", False),
    BenchmarkCase("plausible_but_unsupported", "Ancient Egyptian agriculture", "Why did the inundation matter?",
                  "Ancient Egypt depended on the annual inundation of the Nile River to support its agricultural population. The country's chief wealth came from the fertile floodplain of the Nile valley. Agriculture centered on emmer wheat and barley. The fertility of the land and predictability of the inundation ensured high productivity from a single annual crop, making it possible to store large surpluses.",
                  "The annual Nile flood deposited fertile silt onto the floodplain.", False),
    BenchmarkCase("plausible_but_unsupported", "Ancient Egyptian agriculture", "How did the Nile help farming?",
                  "Ancient Egypt depended on the annual inundation of the Nile River to support its agricultural population. The fertility of the land and predictability of inundation ensured high productivity.",
                  "The Nile flood was the main source of nutrients for Egyptian farms.", False),
    BenchmarkCase("plausible_but_unsupported", "Ancient Egyptian agriculture", "What does the passage say about the flood?",
                  "Ancient Egypt depended on the annual inundation of the Nile River to support its agricultural population. The floodplain was fertile.",
                  "The text describes the annual flood as depositing fertile mud.", False),
    BenchmarkCase("plausible_but_unsupported", "Ancient Egyptian agriculture", "Why did the annual inundation matter?",
                  "Ancient Egypt can be thought of as an oasis in the desert of northeastern Africa, dependent on the annual inundation of the Nile River to support its agricultural population. The country's chief wealth came from the fertile floodplain of the Nile valley. Agriculture centred on emmer wheat and barley. The fertility of the land and general predictability of the inundation ensured very high productivity from a single annual crop. This productivity made it possible to store large surpluses against crop failures.",
                  "The flood, called the annual inundation, deposited a layer of nutrient-rich silt on the floodplain.", False),
]


def main():
    settings = get_settings()
    generator = LearningGenerator(
        InMemoryRetriever(), settings=settings, claim_judge_llm=OllamaProvider(settings),
    )
    results = []
    for index, case in enumerate(CASES, start=1):
        chunk = RetrievedChunk(case.evidence, "semantic_validation_benchmark.txt", index,
                              f"semantic-benchmark-{index}",
                              {"source_filename": "semantic_validation_benchmark.txt", "page": index}, .99)
        context = GroundedContext([chunk], case.evidence, "sufficient", "grounded")
        validation = generator._validate_claim_batch(
            [case.claim], case.topic, case.question, context,
        )
        decision = validation.decisions.get(generator._claim_key(case.claim))
        accepted = bool(decision and decision.accepted and not validation.provider_failed)
        results.append((case, accepted, decision.confidence if decision else None))
        confidence = f"{decision.confidence:.2f}" if decision and decision.confidence is not None else "unavailable"
        print(f"{index:02d} {case.category:27s} expected={'supported' if case.expected_supported else 'rejected':9s} "
              f"actual={'supported' if accepted else 'rejected':9s} confidence={confidence}")

    supported = [row for row in results if row[0].expected_supported]
    unsupported = [row for row in results if not row[0].expected_supported]
    true_accepts = sum(accepted for _, accepted, _ in supported)
    true_rejects = sum(not accepted for _, accepted, _ in unsupported)
    false_positives = [(case, confidence) for case, accepted, confidence in unsupported if accepted]
    false_negatives = [(case, confidence) for case, accepted, confidence in supported if not accepted]

    print(f"\nBenchmark summary ({len(CASES)} cases, temperature 0):")
    print(f"Supported acceptance rate: {true_accepts}/{len(supported)} ({true_accepts / len(supported):.1%})")
    print(f"Unsupported rejection rate: {true_rejects}/{len(unsupported)} ({true_rejects / len(unsupported):.1%})")
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")
    for label, rows in (("False positive", false_positives), ("False negative", false_negatives)):
        for case, confidence in rows:
            print(f"{label}: {case.category}: {case.claim!r} (confidence={confidence})")


if __name__ == "__main__":
    main()
