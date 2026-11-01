import json
from app.generation import LearningGenerator
from app.rag import InMemoryRetriever
from app.domain import RetrievedChunk
from app.schemas import MCQRequest, LearnerProfileIn

def run():
    # This is a small deterministic wiring check, not a retrieval benchmark or quality metric.
    r = InMemoryRetriever([RetrievedChunk("CNN convolution extracts local spatial features.", "sample.txt", 1, "c1")])
    g = LearningGenerator(r); low = g.mcq(MCQRequest(topic="CNN", number_of_questions=2, learner_profile=LearnerProfileIn(topic_mastery={"CNN": .31}, weak_topics=["CNN"])))
    return {"schema_valid": True, "grounded": low.grounding.grounding_status == "grounded", "adapted_difficulty": low.adaptation.adapted_settings["difficulty"]}
if __name__ == "__main__": print(json.dumps(run(), indent=2))
