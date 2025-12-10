import json
from dataclasses import dataclass
from typing import Dict, List
from .together_client import complete, acomplete
from .teacher import TeacherGuess
from .student import StudentProfile

JUDGE_MODEL = "moonshotai/Kimi-K2-Thinking"


@dataclass
class JudgeResult:
    accuracy: float
    correct: List[str]
    incorrect: List[str]
    details: Dict[str, dict]


class Judge:
    def __init__(self, ground_truth: StudentProfile, model: str = JUDGE_MODEL):
        self.ground_truth = ground_truth
        self.model = model

    def _build_prompt(self, guess: TeacherGuess) -> str:
        ground_truth_str = json.dumps(self.ground_truth.knowledge_state)

        guess_dict = {}
        for e in guess.estimates:
            guess_dict[e.topic] = {"known": e.known, "confidence": e.confidence, "reasoning": e.reasoning}
        guess_str = json.dumps(guess_dict)

        return f"""Compare the teacher's guess against the ground truth.

GROUND TRUTH (what student actually knows):
{ground_truth_str}

TEACHER'S GUESS:
{guess_str}

For each topic, determine if the teacher's guess matches the ground truth.

Respond with JSON only:
{{
  "results": [
    {{"topic": "name", "actual": true/false, "guessed": true/false, "correct": true/false}}
  ]
}}"""

    def _parse_response(self, response_text: str, guess: TeacherGuess) -> JudgeResult:
        try:
            text = response_text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text)

            correct = []
            incorrect = []
            details = {}

            for r in data["results"]:
                topic = r["topic"]
                details[topic] = {
                    "actual": r["actual"],
                    "guessed": r["guessed"],
                    "correct": r["correct"],
                }
                if r["correct"]:
                    correct.append(topic)
                else:
                    incorrect.append(topic)

            total = len(correct) + len(incorrect)
            accuracy = len(correct) / total if total > 0 else 0.0

            return JudgeResult(accuracy=accuracy, correct=correct, incorrect=incorrect, details=details)

        except (json.JSONDecodeError, KeyError):
            # Fallback to deterministic comparison
            return self._deterministic_grade(guess)

    def _deterministic_grade(self, guess: TeacherGuess) -> JudgeResult:
        correct = []
        incorrect = []
        details = {}

        for estimate in guess.estimates:
            topic = estimate.topic
            if topic not in self.ground_truth.knowledge_state:
                continue

            actual = self.ground_truth.knowledge_state[topic]
            guessed = estimate.known

            details[topic] = {
                "actual": actual,
                "guessed": guessed,
                "correct": actual == guessed,
            }

            if actual == guessed:
                correct.append(topic)
            else:
                incorrect.append(topic)

        total = len(correct) + len(incorrect)
        accuracy = len(correct) / total if total > 0 else 0.0

        return JudgeResult(accuracy=accuracy, correct=correct, incorrect=incorrect, details=details)

    def grade(self, guess: TeacherGuess, temperature: float = 0.1) -> JudgeResult:
        response = complete(
            self._build_prompt(guess),
            model=self.model,
            temperature=temperature,
        )
        return self._parse_response(response.content, guess)

    async def agrade(self, guess: TeacherGuess, temperature: float = 0.1) -> JudgeResult:
        response = await acomplete(
            self._build_prompt(guess),
            model=self.model,
            temperature=temperature,
        )
        return self._parse_response(response.content, guess)
