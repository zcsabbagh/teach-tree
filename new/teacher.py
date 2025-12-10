import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from .together_client import complete, acomplete
from .parsing import (
    parse_assessment_response,
    regex_extract_estimates,
    build_llm_extraction_prompt,
    fill_missing_topics,
)
from .utils import extract_number_from_text

TEACHER_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"


@dataclass
class KnowledgeEstimate:
    topic: str
    known: bool
    confidence: float  # 0-1
    reasoning: str


@dataclass
class TeacherGuess:
    estimates: List[KnowledgeEstimate]
    raw_response: str

    def to_dict(self) -> Dict[str, bool]:
        return {e.topic: e.known for e in self.estimates}


@dataclass
class ConversationTurn:
    question: str
    answer: str


@dataclass
class QuestionResult:
    question: str
    candidates: Optional[List[str]] = None
    selected_idx: Optional[int] = None


class SimulatedTeacher:
    def __init__(
        self,
        subject: str,
        topics: List[str],
        model: str = TEACHER_MODEL,
    ):
        self.subject = subject
        self.topics = topics
        self.model = model
        self.history: List[ConversationTurn] = []
        self.last_estimate: Optional[TeacherGuess] = None

    def _build_system_prompt(self) -> str:
        topics_str = ", ".join(self.topics)
        return f"""You are a teacher assessing a student's knowledge in {self.subject}.

TOPICS TO ASSESS: {topics_str}

You must respond with valid JSON only:
{{
  "estimates": [
    {{"topic": "topic_name", "known": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
  ]
}}

HOW TO ASSESS:

MARK known=true IF:
- Student gets the CORRECT answer with correct reasoning
- Student demonstrates understanding (correct formulas, methods, terminology)
- Calculations are correct

MARK known=false IF:
- Student gets the WRONG answer
- Student makes calculation errors
- Student misidentifies concepts (e.g., calls skewed data "normal")
- Student expresses confusion ("I don't know", "I'm not sure how to do this")
- No evidence for this topic yet

IMPORTANT DISTINCTIONS:
- Correct calculation + correct answer = known=true
- Correct method + calculation error = known=false (wrong final answer)
- "I don't understand this concept" = known=false
- Confident wrong answer = known=false
- No mention of topic = known=false (confidence: 0.5)

Base your assessment on what the student DEMONSTRATED, not assumptions."""

    def _build_assessment_prompt(self) -> str:
        history_str = ""
        for i, turn in enumerate(self.history, 1):
            history_str += f"\n[Turn {i}]\nTeacher: {turn.question}\nStudent: {turn.answer}\n"

        return f"""Conversation so far:{history_str}

Based on this conversation, assess the student's knowledge of each topic: {", ".join(self.topics)}

Respond with JSON only."""

    def _dicts_to_estimates(self, estimate_dicts: List[Dict[str, Any]]) -> List[KnowledgeEstimate]:
        """Convert list of dicts to KnowledgeEstimate objects."""
        return [
            KnowledgeEstimate(
                topic=e["topic"],
                known=e["known"],
                confidence=e["confidence"],
                reasoning=e["reasoning"],
            )
            for e in estimate_dicts
        ]

    def _parse_response(self, response_text: str) -> TeacherGuess:
        """Parse assessment response - uses LLM extraction for reliability."""
        estimates, missing = parse_assessment_response(response_text, self.topics)

        if not missing:
            return TeacherGuess(
                estimates=self._dicts_to_estimates(estimates),
                raw_response=response_text,
            )

        # Use LLM to extract structured assessment - this is the reliable fallback
        return self._llm_extract_assessment(response_text)

    def _llm_extract_assessment(self, raw_response: str) -> TeacherGuess:
        """Use LLM to extract structured assessment from any response format."""
        prompt = build_llm_extraction_prompt(raw_response, self.topics)
        response = complete(prompt, model=self.model, temperature=0.1, max_tokens=500)
        return self._parse_llm_extraction(response.content, raw_response)

    async def _allm_extract_assessment(self, raw_response: str) -> TeacherGuess:
        """Async version: Use LLM to extract structured assessment."""
        prompt = build_llm_extraction_prompt(raw_response, self.topics)
        response = await acomplete(prompt, model=self.model, temperature=0.1, max_tokens=500)
        return self._parse_llm_extraction(response.content, raw_response)

    def _parse_llm_extraction(self, extraction_response: str, original_response: str) -> TeacherGuess:
        """Parse the LLM extraction response into TeacherGuess with multiple fallbacks."""
        estimates, missing = parse_assessment_response(extraction_response, self.topics)

        # Try regex fallback on both responses for missing topics
        if missing:
            combined_text = f"{original_response}\n{extraction_response}"
            regex_estimates = regex_extract_estimates(combined_text, missing)
            estimates.extend(regex_estimates)

        # Fill in any remaining missing topics with defaults
        estimates = fill_missing_topics(estimates, self.topics)

        return TeacherGuess(
            estimates=self._dicts_to_estimates(estimates),
            raw_response=original_response,
        )

    def record_turn(self, question: str, answer: str) -> None:
        self.history.append(ConversationTurn(question=question, answer=answer))

    def assess(self, temperature: float = 0.3) -> TeacherGuess:
        if not self.history:
            # No evidence yet
            estimates = [
                KnowledgeEstimate(topic=t, known=False, confidence=0.5, reasoning="No evidence yet")
                for t in self.topics
            ]
            return TeacherGuess(estimates=estimates, raw_response="")

        response = complete(
            self._build_assessment_prompt(),
            model=self.model,
            temperature=temperature,
            system_prompt=self._build_system_prompt(),
        )
        return self._parse_response(response.content)

    async def aassess(self, temperature: float = 0.3) -> TeacherGuess:
        if not self.history:
            estimates = [
                KnowledgeEstimate(topic=t, known=False, confidence=0.5, reasoning="No evidence yet")
                for t in self.topics
            ]
            return TeacherGuess(estimates=estimates, raw_response="")

        response = await acomplete(
            self._build_assessment_prompt(),
            model=self.model,
            temperature=temperature,
            system_prompt=self._build_system_prompt(),
        )
        self.last_estimate = await self._aparse_response(response.content)
        return self.last_estimate

    async def _aparse_response(self, response_text: str) -> TeacherGuess:
        """Async parse assessment response - uses LLM extraction for reliability."""
        estimates, missing = parse_assessment_response(response_text, self.topics)

        if not missing:
            return TeacherGuess(
                estimates=self._dicts_to_estimates(estimates),
                raw_response=response_text,
            )

        # Use async LLM extraction as fallback
        return await self._allm_extract_assessment(response_text)

    def generate_question(self, target_topic: Optional[str] = None, temperature: float = 0.7) -> str:
        """Generate a question to probe student knowledge."""
        if target_topic:
            prompt = f"Generate a question to test if a student understands '{target_topic}' in {self.subject}. Just the question, nothing else."
        else:
            prompt = f"Generate a question to test student knowledge in {self.subject}, focusing on one of these topics: {', '.join(self.topics)}. Just the question, nothing else."

        response = complete(prompt, model=self.model, temperature=temperature, max_tokens=200)
        return response.content.strip()

    def _get_uncertain_topics(self) -> List[str]:
        """Return topics with confidence < 0.8, prioritizing lowest confidence."""
        if not self.last_estimate:
            return self.topics
        uncertain = [(e.topic, e.confidence) for e in self.last_estimate.estimates if e.confidence < 0.8]
        if not uncertain:
            return self.topics
        uncertain.sort(key=lambda x: x[1])
        return [t for t, _ in uncertain]

    async def agenerate_question(self, target_topic: Optional[str] = None, temperature: float = 0.7, n: int = 1) -> QuestionResult:
        # Target uncertain topics
        uncertain = self._get_uncertain_topics()
        target = target_topic or (uncertain[0] if uncertain else None)

        if n == 1:
            q = await self._agenerate_single_question(target, temperature)
            return QuestionResult(question=q)

        # Generate N candidates targeting different uncertain topics
        targets = (uncertain * n)[:n]  # Cycle through uncertain topics
        candidates = await asyncio.gather(*[
            self._agenerate_single_question(t, temperature) for t in targets
        ])
        candidates = list(candidates)

        question, idx = await self._aselect_best_question(candidates, uncertain)
        return QuestionResult(question=question, candidates=candidates, selected_idx=idx)

    async def _agenerate_single_question(self, target_topic: Optional[str] = None, temperature: float = 0.7) -> str:
        # Build context from conversation history
        context = ""
        if self.history:
            context = "Previous exchanges:\n"
            for turn in self.history[-3:]:  # Last 3 turns for context
                context += f"Q: {turn.question}\nA: {turn.answer}\n\n"

        # Include current knowledge estimates
        estimates_str = ""
        if self.last_estimate:
            estimates_str = "Current assessment:\n"
            for e in self.last_estimate.estimates:
                status = "KNOWS" if e.known else "DOESN'T KNOW"
                estimates_str += f"- {e.topic}: {status} (confidence: {e.confidence:.0%})\n"

        if target_topic:
            prompt = f"""{context}{estimates_str}
Generate a short question that tests '{target_topic}' in {self.subject}.
The question should help determine if the student truly understands this topic.
Avoid questions similar to ones already asked.
Just output the question, nothing else."""
        else:
            prompt = f"""{context}{estimates_str}
Generate a question to test student knowledge in {self.subject}, focusing on: {', '.join(self.topics)}.
Just output the question, nothing else."""

        response = await acomplete(prompt, model=self.model, temperature=temperature, max_tokens=200)
        return response.content.strip()

    async def _aselect_best_question(self, candidates: List[str], uncertain_topics: List[str]) -> tuple[str, int]:
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(candidates))

        # Build detailed assessment state with reasoning
        assessment_str = "CURRENT KNOWLEDGE ASSESSMENT:\n"
        if self.last_estimate:
            for e in self.last_estimate.estimates:
                # Calculate uncertainty: closer to 50% = more uncertain
                uncertainty = 1 - abs(e.confidence - 0.5) * 2  # 0 at 0% or 100%, 1 at 50%
                status = "KNOWS" if e.known else "DOESN'T KNOW"
                assessment_str += f"- {e.topic}: {status} (confidence: {e.confidence:.0%}, uncertainty: {uncertainty:.0%})\n"
                assessment_str += f"  Reasoning: {e.reasoning}\n"
        else:
            assessment_str += "No assessment yet - all topics equally uncertain.\n"

        # Include conversation context
        context = ""
        if self.history:
            context = "\nPREVIOUS EXCHANGES:\n"
            for turn in self.history[-2:]:
                context += f"Q: {turn.question}\nA: {turn.answer[:150]}...\n\n"

        prompt = f"""{assessment_str}{context}
GOAL: Select the question that will DECREASE UNCERTAINTY THE MOST.

Key principles:
- Questions about topics with confidence near 50% are most valuable (high uncertainty)
- Questions about topics with confidence near 0% or 100% are less valuable (already certain)
- A good question tests ONE topic cleanly, not multiple topics at once
- Avoid asking about topics we're already confident about

CANDIDATE QUESTIONS:
{numbered}

For each candidate, briefly consider:
- What topic does it primarily test?
- How uncertain are we about that topic?
- Will the answer clearly confirm or deny knowledge?

Then output ONLY the number of the best question (1-{len(candidates)})."""

        response = await acomplete(prompt, model=self.model, temperature=0.1, max_tokens=100)

        # Extract the number from the response
        idx = extract_number_from_text(response.content, len(candidates))
        if idx is not None:
            return candidates[idx], idx
        return candidates[0], 0
