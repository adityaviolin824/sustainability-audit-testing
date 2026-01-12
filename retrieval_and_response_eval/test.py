import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")

# WE JUST LOAD ALL THE QUESTIONS FROM THE JSONL FILE >>>>HAVE TO CUSTOMIZE ACCORDING TO USE CASE<<<< AND THEN WE RETURN IT IN THE FORM OF A LIST, SO THAT TESTING CAN BE PERFORMED

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
            if not line:  # This skips the 'space in between' and empty lines at the end
                continue
            
            try:
                data = json.loads(line)
                tests.append(TestQuestion(**data))
            except json.JSONDecodeError as e:
                # Log the error but keep going so one bad line doesn't kill the whole eval
                print(f"Skipping malformed JSON line: {e}")
                continue
                
    return tests
