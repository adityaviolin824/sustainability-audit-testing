import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")

class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests() -> list[TestQuestion]:
    """Load test questions from JSONL file, skipping empty lines."""
    tests = []
    if not Path(TEST_FILE).exists():
        print(f"Error: Test file not found at {TEST_FILE}")
        return []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            
            try:
                data = json.loads(line)
                tests.append(TestQuestion(**data))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line: {e}")
                continue
                
    return tests
