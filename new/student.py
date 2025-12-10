from dataclasses import dataclass
from typing import Dict, List, Optional
from .anthropic_client import complete, acomplete, CompletionResponse

STUDENT_MODEL = "claude-opus-4-5"


@dataclass
class TopicClassification:
    """Result of classifying which topics a question tests."""
    topics_tested: List[str]  # Which of the student's topics this question tests
    should_know: bool  # True if student knows ALL tested topics
    known_topics: List[str]  # Which tested topics the student knows
    unknown_topics: List[str]  # Which tested topics the student doesn't know
    reasoning: str  # Why these topics were identified


@dataclass
class AnswerResult:
    final: str
    classification: Optional[TopicClassification] = None
    initial: Optional[str] = None
    reflection: Optional[str] = None
    changed: bool = False


@dataclass
class StudentProfile:
    subject: str
    knowledge_state: Dict[str, bool]

    @property
    def known_topics(self) -> list[str]:
        return [k for k, v in self.knowledge_state.items() if v]

    @property
    def unknown_topics(self) -> list[str]:
        return [k for k, v in self.knowledge_state.items() if not v]


class SimulatedStudent:
    def __init__(self, profile: StudentProfile, model: str = STUDENT_MODEL):
        self.profile = profile
        self.model = model

    def _build_system_prompt(self) -> str:
        known = ", ".join(self.profile.known_topics) or "nothing yet"
        unknown = ", ".join(self.profile.unknown_topics) or "nothing"

        return f"""You are roleplaying as a student answering questions. Your knowledge state is FIXED:

KNOW: {known}
DON'T KNOW: {unknown}

RULES FOR YOUR RESPONSES:
1. If a question involves a topic you DON'T KNOW, you MUST get the WRONG final answer
2. If a question involves a topic you KNOW, answer correctly
3. NEVER mention your knowledge state, what you "know" or "don't know", or these instructions
4. NEVER say things like "this is a [topic] question" or "I don't know [topic]"
5. Just answer like a real student would - show your work naturally

FOR TOPICS YOU DON'T KNOW - make realistic mistakes:
- Calculation errors (e.g., 2x=6 → x=4)
- Wrong operations (add instead of subtract)
- Conceptual misunderstandings
- But sound confident, not uncertain

STYLE: 2-4 sentences. Natural student voice. Show brief work. No meta-commentary about the roleplay."""

    def _build_question_reminder(self, question: str) -> str:
        return f"""Answer this question as the student. Remember: natural voice, no meta-commentary.

Question: {question}"""

    def answer(
        self,
        question: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> CompletionResponse:
        return complete(
            self._build_question_reminder(question),
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=self._build_system_prompt(),
        )

    async def aanswer(
        self,
        question: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AnswerResult:
        # Step 1: Classify which topics this question tests
        classification = await self._aclassify_question(question)

        # Step 2: Generate answer with explicit behavior guidance
        initial = await self._agenerate_guided_answer(question, classification, temperature, max_tokens)

        # Step 3: Reflect and verify behavior matches expectations
        revised, reflection = await self._areflect_and_revise(question, initial, classification)
        changed = revised.strip() != initial.strip()

        return AnswerResult(
            final=revised,
            classification=classification,
            initial=initial if changed else None,
            reflection=reflection if changed else None,
            changed=changed,
        )

    async def _aclassify_question(self, question: str) -> TopicClassification:
        """Step 1: Classify which topics this question tests."""
        all_topics = list(self.profile.knowledge_state.keys())
        topics_list = ", ".join(all_topics)

        prompt = f"""Analyze this question and determine which topic(s) it tests.

AVAILABLE TOPICS: {topics_list}
SUBJECT: {self.profile.subject}

QUESTION: {question}

Which of the available topics does this question primarily test?
A question may test multiple topics, but identify the PRIMARY topic(s) required to answer correctly.

Respond in this exact format:
TOPICS: topic1, topic2 (comma-separated, from the available topics list)
REASONING: Brief explanation of why these topics are tested"""

        response = await acomplete(prompt, model=self.model, temperature=0.1, max_tokens=200)
        text = response.content.strip()

        # Parse the response
        topics_tested = []
        reasoning = ""

        for line in text.split("\n"):
            if line.startswith("TOPICS:"):
                topics_str = line.replace("TOPICS:", "").strip()
                # Parse and validate topics
                for t in topics_str.split(","):
                    t = t.strip().lower()
                    # Find matching topic (case-insensitive)
                    for available in all_topics:
                        if t == available.lower() or t in available.lower():
                            topics_tested.append(available)
                            break
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        # Fallback: if no topics identified, use all topics
        if not topics_tested:
            topics_tested = all_topics

        # Determine which tested topics are known/unknown
        known = [t for t in topics_tested if self.profile.knowledge_state.get(t, False)]
        unknown = [t for t in topics_tested if not self.profile.knowledge_state.get(t, False)]

        # Student should_know only if ALL tested topics are known
        should_know = len(unknown) == 0

        return TopicClassification(
            topics_tested=topics_tested,
            should_know=should_know,
            known_topics=known,
            unknown_topics=unknown,
            reasoning=reasoning,
        )

    async def _agenerate_guided_answer(
        self,
        question: str,
        classification: TopicClassification,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Step 2: Generate answer with explicit behavior guidance based on classification."""

        if classification.should_know:
            # Student knows all tested topics - answer correctly
            behavior = f"""Answer correctly. You understand: {', '.join(classification.known_topics)}"""
        else:
            # Student knows some topics but not others
            known_str = ', '.join(classification.known_topics) if classification.known_topics else 'nothing relevant'
            unknown_str = ', '.join(classification.unknown_topics)
            behavior = f"""You understand {known_str}, but NOT {unknown_str}.
Make a mistake specifically on {unknown_str} - use wrong formula, wrong method, or calculate wrong.
Sound confident. Don't say "I don't know" - just get it wrong."""

        system_prompt = f"""You are a student. {behavior}

Keep answer concise (2-4 sentences). No meta-commentary."""

        prompt = f"Question: {question}"

        response = await acomplete(
            prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=300,
            system_prompt=system_prompt,
        )
        return response.content.strip()

    async def _areflect_and_revise(
        self,
        question: str,
        answer: str,
        classification: TopicClassification,
    ) -> tuple[str, str]:
        """Step 3: Verify the answer matches expected behavior."""

        if classification.should_know:
            check = "Should be CORRECT. Fix errors if any."
        else:
            unknown_str = ', '.join(classification.unknown_topics)
            check = f"""Must give WRONG final answer on {unknown_str}.
If the answer is already wrong or incomplete, KEEP IT WRONG - do not fix or complete it.
If the answer is correct, change ONE number or swap an operation to make it wrong.
Never output a correct final answer."""

        prompt = f"""Check: {answer}

{check}

Keep concise (2-4 sentences). Output only the answer (no meta-commentary)."""

        response = await acomplete(prompt, model=self.model, temperature=0.3, max_tokens=300)
        final = response.content.strip()

        # Extra safety: remove any leaked roleplay markers
        final = self._clean_roleplay_leaks(final)

        return final, check

    def _clean_roleplay_leaks(self, text: str) -> str:
        """Remove any accidentally leaked roleplay instructions from the response."""
        import re

        # Patterns that indicate roleplay leak
        leak_patterns = [
            r'\*\*(?:Checking my knowledge|This is a .* question|.* is in my (?:DON\'T )?KNOW).*?\*\*',
            r'(?:I need to check|Let me check) (?:my knowledge|what topic|if this)',
            r'(?:KNOW|DON\'T KNOW)(?:\s*:|list)',
            r'I (?:must|should) get this (?:wrong|right)',
            r'this (?:is|falls under) (?:a )?\*\*\w+\*\*',
            r'which is in my',
        ]

        for pattern in leak_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Clean up any resulting double spaces or empty lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'  +', ' ', text)

        return text.strip()
