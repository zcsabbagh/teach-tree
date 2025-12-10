"""
Inverse Cognitive Search: Test-Time Compute for Student Diagnosis

A diagnostic system that identifies a student's hidden skill profile using
LLM-powered question generation and scoring. Tests how increasing the number
of candidate questions (N) improves diagnostic efficiency.
"""
from __future__ import annotations

import json
import random
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from io import StringIO
from typing import Optional, Union, List, Dict
from dotenv import load_dotenv
from together import Together

from student_profiles import (
    SkillProfile,
    SAMPLE_PROFILES,
    get_sample_profile,
    list_sample_profiles,
    get_profile_names,
    LEVELS,
    SKILL_DOMAINS,
    set_skill_domain,
    set_custom_skills,
    get_current_skills,
)


# Load environment variables from .env.local
load_dotenv(".env.local")

# Initialize Together client
client = Together()

# Constants
MAX_TURNS = 10
CONFIDENCE_THRESHOLD = 0.9
JSON_PARSE_RETRIES = 3

# Model configuration
# Student uses stronger model for better role-play fidelity
STUDENT_MODEL = "moonshotai/Kimi-K2-Thinking"
# Teacher uses smaller model (the one being tested)
TEACHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# Retry configuration
API_MAX_RETRIES = 5
API_BASE_DELAY = 1.0  # seconds
API_MAX_DELAY = 60.0  # seconds


# =============================================================================
# LLM Interface
# =============================================================================

class APIError(Exception):
    """Custom exception for API errors."""
    pass


def llm_call(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    model: str = TEACHER_MODEL
) -> str:
    """Make a single LLM call to Together API with retry and rate limiting."""
    last_error = None

    for attempt in range(API_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check for rate limiting errors
            is_rate_limit = any(phrase in error_str for phrase in [
                "rate limit", "rate_limit", "429", "too many requests",
                "quota", "throttl"
            ])

            # Check for transient/server errors
            is_transient = any(phrase in error_str for phrase in [
                "500", "502", "503", "504", "server error", "timeout",
                "connection", "temporary", "unavailable"
            ])

            if is_rate_limit or is_transient:
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    API_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1),
                    API_MAX_DELAY
                )

                error_type = "Rate limited" if is_rate_limit else "Transient error"
                print(f"\n  [{error_type}] Retrying in {delay:.1f}s (attempt {attempt + 1}/{API_MAX_RETRIES})...")
                time.sleep(delay)
                continue
            else:
                # Non-retryable error
                raise APIError(f"API call failed: {e}") from e

    raise APIError(f"API call failed after {API_MAX_RETRIES} retries: {last_error}")


def parse_json_from_response(response: str) -> Union[dict, list]:
    """Extract and parse JSON from LLM response, handling ```json blocks."""
    # Try to find JSON in code blocks first
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end != -1:
            json_str = response[start:end].strip()
            return json.loads(json_str)

    # Try to find JSON in generic code blocks
    if "```" in response:
        start = response.find("```") + 3
        # Skip language identifier if present
        newline = response.find("\n", start)
        if newline != -1:
            start = newline + 1
        end = response.find("```", start)
        if end != -1:
            json_str = response[start:end].strip()
            return json.loads(json_str)

    # Try to parse the whole response as JSON
    # Find the first [ or { and parse from there
    for i, char in enumerate(response):
        if char in "[{":
            # Find matching closing bracket
            try:
                return json.loads(response[i:])
            except json.JSONDecodeError:
                continue

    # Last resort: try the whole thing
    return json.loads(response)


def llm_call_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    retries: int = JSON_PARSE_RETRIES
) -> Union[dict, list]:
    """Make an LLM call and parse JSON response with retry logic."""
    last_error = None
    for attempt in range(retries):
        try:
            response = llm_call(system_prompt, user_prompt, temperature)
            return parse_json_from_response(response)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < retries - 1:
                continue
    raise ValueError(f"Failed to parse JSON after {retries} attempts: {last_error}")


# =============================================================================
# Data Structures
# =============================================================================

def random_profile(seed: Optional[int] = None, domain: Optional[str] = None) -> SkillProfile:
    """Generate a random skill profile from a random or specified domain."""
    if seed is not None:
        random.seed(seed)

    # Pick a random domain if not specified
    if domain is None:
        domain = random.choice(list(SKILL_DOMAINS.keys()))

    skills = SKILL_DOMAINS[domain]
    levels = {skill: random.choice(LEVELS) for skill in skills}
    return SkillProfile(levels=levels, domain=domain)


@dataclass
class BeliefState:
    """Probability distributions over skill levels for each skill."""
    distributions: dict[str, list[float]]  # skill -> [p_low, p_med, p_high]

    @classmethod
    def uniform(cls) -> "BeliefState":
        """Initialize with uniform distributions."""
        # Binary levels: [p_low, p_high]
        return cls(distributions={
            skill: [0.5, 0.5] for skill in get_current_skills()
        })

    def get_confidence(self, skill: str) -> float:
        """Get the maximum probability for a skill."""
        return max(self.distributions[skill])

    def get_prediction(self, skill: str) -> str:
        """Get the predicted level (argmax) for a skill."""
        probs = self.distributions[skill]
        idx = probs.index(max(probs))
        return LEVELS[idx]

    def all_confident(self, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
        """Check if all skills have reached confidence threshold."""
        return all(
            self.get_confidence(skill) >= threshold
            for skill in get_current_skills()
        )

    def __str__(self) -> str:
        lines = []
        for skill in get_current_skills():
            probs = self.distributions[skill]
            pred = self.get_prediction(skill)
            conf = self.get_confidence(skill)
            # Binary levels: [low, high]
            lines.append(
                f"  {skill}: L={probs[0]:.2f} H={probs[1]:.2f} "
                f"-> {pred} ({conf:.0%})"
            )
        return "\n".join(lines)


@dataclass
class ConversationHistory:
    """Stores the Q&A history of a diagnostic session."""
    turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, question: str, answer: str):
        self.turns.append((question, answer))

    def format_for_prompt(self) -> str:
        if not self.turns:
            return "No conversation history yet."

        lines = []
        for i, (q, a) in enumerate(self.turns, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  Q: {q}")
            lines.append(f"  A: {a}")
        return "\n".join(lines)


# =============================================================================
# Dynamic Skill Knowledge Generator
# =============================================================================

# Cache for generated knowledge to avoid redundant LLM calls
_skill_knowledge_cache: Dict[str, Dict[str, dict]] = {}


def generate_skill_knowledge(skill: str, level: str) -> dict:
    """Generate knowledge/misconceptions for a skill at a given level using LLM.

    This allows the system to work with ANY skill domain - CS, math, science, etc.
    Results are cached to avoid redundant generation.
    """
    # Check cache first
    cache_key = f"{skill}:{level}"
    if skill in _skill_knowledge_cache and level in _skill_knowledge_cache[skill]:
        return _skill_knowledge_cache[skill][level]

    system_prompt = """You are an expert educator creating a student knowledge profile.
Generate SPECIFIC facts, examples, and behaviors for a student at a given skill level.

Output valid JSON with this EXACT structure:
{
    "facts": ["fact 1", "fact 2", ...],
    "examples": ["example 1", "example 2", ...],
    "behavior": "description of how student should respond"
}

For HIGH level: correct facts, working examples, confident behavior
For LOW level: misconceptions (WRONG facts), broken examples, confused behavior"""

    if level == "high":
        user_prompt = f"""Generate knowledge for a student with HIGH skill in "{skill}".

Include:
- 4-5 CORRECT facts they know well
- 2-3 CORRECT examples (code, formulas, or explanations depending on domain)
- Behavior: confident, uses proper terminology

Output JSON only:"""

    else:  # low
        user_prompt = f"""Generate knowledge for a student with LOW skill in "{skill}".

Include:
- 4-5 MISCONCEPTIONS (common WRONG beliefs about this topic)
- 2-3 INCORRECT examples (buggy code, wrong formulas, flawed explanations)
- Behavior: confused, uses wrong terminology, hedges with "I think maybe..."

Output JSON only:"""

    try:
        result = llm_call_json(system_prompt, user_prompt, temperature=0.7)

        # Ensure required keys exist
        if not isinstance(result, dict):
            result = {}
        result.setdefault("facts", [])
        result.setdefault("examples", [])
        result.setdefault("behavior", "Respond according to skill level.")

        # Cache the result
        if skill not in _skill_knowledge_cache:
            _skill_knowledge_cache[skill] = {}
        _skill_knowledge_cache[skill][level] = result

        return result

    except Exception as e:
        # Fallback if generation fails
        return {
            "facts": [f"Basic understanding of {skill}" if level != "low" else f"Confused about {skill}"],
            "examples": [],
            "behavior": "Answer according to skill level."
        }


# =============================================================================
# StudentSimulator
# =============================================================================

class StudentSimulator:
    """Simulates a student with a hidden skill profile."""

    def __init__(self, profile: SkillProfile):
        self.profile = profile

    def _build_skill_knowledge(self, skill: str, level: str) -> str:
        """Build detailed knowledge section for a skill at given level.

        Uses dynamic LLM generation - works for ANY skill domain.
        Binary levels: low or high only.
        """
        # Generate knowledge dynamically (cached)
        knowledge = generate_skill_knowledge(skill, level)

        lines = [f"\n### {skill.upper()} (your level: {level.upper()})"]

        if level == "high":
            lines.append("FACTS YOU KNOW (use these in your answers):")
            for fact in knowledge.get("facts", []):
                lines.append(f"  - {fact}")
            lines.append("EXAMPLES YOU CAN USE CORRECTLY:")
            for example in knowledge.get("examples", []):
                lines.append(f"  - {example}")
            lines.append(f"HOW TO BEHAVE: {knowledge.get('behavior', 'Answer confidently and correctly.')}")

        else:  # low
            lines.append("YOUR MISCONCEPTIONS (use these WRONG ideas):")
            for fact in knowledge.get("facts", []):
                lines.append(f"  - {fact}")
            lines.append("WRONG EXAMPLES YOU WOULD GIVE:")
            for example in knowledge.get("examples", []):
                lines.append(f"  - {example}")
            lines.append(f"HOW TO BEHAVE: {knowledge.get('behavior', 'Be confused and make errors.')}")

        return "\n".join(lines)

    def answer(self, question: str, history: Optional[ConversationHistory] = None) -> str:
        """Generate a realistic student response based on skill levels.

        Args:
            question: The question to answer
            history: Previous conversation history for consistency
        """
        # Build detailed knowledge for each skill
        knowledge_sections = []
        for skill, level in self.profile.levels.items():
            knowledge_sections.append(self._build_skill_knowledge(skill, level))

        knowledge_text = "\n".join(knowledge_sections)

        # Build clear skill summary
        skill_summary_lines = []
        for skill, level in self.profile.levels.items():
            if level == "high":
                skill_summary_lines.append(f"  - {skill}: HIGH (you are GOOD at this - answer correctly)")
            else:
                skill_summary_lines.append(f"  - {skill}: LOW (you are BAD at this - give WRONG answers)")
        skill_summary = "\n".join(skill_summary_lines)

        system_prompt = f"""You are roleplaying as a student with these EXACT skill levels:

############################################
# YOUR SKILL LEVELS - FOLLOW THESE EXACTLY #
############################################
{skill_summary}

=== DETAILED KNOWLEDGE ===
{knowledge_text}

############################################
# MANDATORY RULES - YOU MUST OBEY THESE    #
############################################

STEP 1: Identify which skill the question tests
STEP 2: Check YOUR level for that skill above
STEP 3: Respond EXACTLY according to your level:

FOR LOW SKILLS (you MUST do ALL of these):
   ❌ Give WRONG answers - incorrect facts, flawed logic
   ❌ Show confusion - "I think maybe...", "I'm not sure...", "I don't really get..."
   ❌ Make mistakes - wrong terminology, incorrect examples
   ⚠️ NEVER accidentally give a correct answer for LOW skills

FOR HIGH SKILLS (you MUST do ALL of these):
   ✓ Give CORRECT answers - accurate facts, sound logic
   ✓ Show confidence - no hedging, no "maybe"
   ✓ Use proper terminology from your knowledge

CRITICAL: Your skill level is FIXED. Do not change behavior between questions.
If you are LOW on a skill, you are ALWAYS confused about it.
If you are HIGH on a skill, you ALWAYS know it well.

Keep responses to 2-4 sentences. Never mention you're roleplaying."""

        # Include conversation history for consistency
        if history and history.turns:
            history_text = "\n\nPREVIOUS CONVERSATION:\n"
            history_text += history.format_for_prompt()
        else:
            history_text = ""

        # Remind of skills in user prompt too with strong enforcement
        user_prompt = f"""{history_text}

###########################################
# BEFORE YOU ANSWER - CHECK YOUR SKILLS:  #
###########################################
{skill_summary}

QUESTION: {question}

⚠️ STOP: Which skill does this test? What is YOUR level? Answer accordingly:"""

        # Use stronger model for student to ensure faithful role-play
        # Lower temperature (0.5) for more consistent behavior
        return llm_call(system_prompt, user_prompt, temperature=0.5, model=STUDENT_MODEL)


# =============================================================================
# QuestionGenerator
# =============================================================================

class QuestionGenerator:
    """Generates candidate diagnostic questions."""

    def generate(
        self,
        n: int,
        beliefs: BeliefState,
        history: ConversationHistory
    ) -> list[str]:
        """Generate N diverse candidate questions."""

        # Find uncertain skills (those below threshold)
        uncertain_skills = [
            skill for skill in get_current_skills()
            if beliefs.get_confidence(skill) < CONFIDENCE_THRESHOLD
        ]

        # Build dynamic skill list
        current_skills = get_current_skills()
        skills_text = ", ".join(current_skills)

        system_prompt = f"""You are a diagnostic question generator.
Generate diverse questions that help identify a student's skill level.

The skills to probe are: {skills_text}

Good diagnostic questions:
- Target ONE specific skill (not multiple)
- Can reveal low vs high proficiency based on the answer
- Vary in difficulty and approach
- Don't repeat similar questions from history

Output ONLY a JSON array of question strings, no other text.
Format: ["question 1", "question 2", ...]"""

        beliefs_summary = "\n".join(
            f"- {skill}: {beliefs.get_prediction(skill)} ({beliefs.get_confidence(skill):.0%} confident)"
            for skill in get_current_skills()
        )

        user_prompt = f"""Generate exactly {n} diverse diagnostic questions.

Current beliefs about student:
{beliefs_summary}

Skills still uncertain (below 90% confidence): {', '.join(uncertain_skills) if uncertain_skills else 'None'}

Conversation history:
{history.format_for_prompt()}

Focus questions on uncertain skills. Generate {n} questions as a JSON array:"""

        questions = llm_call_json(system_prompt, user_prompt, temperature=0.7)

        # Ensure we have exactly N questions
        if isinstance(questions, list):
            if len(questions) < n:
                # Pad with generic questions for current skills
                while len(questions) < n:
                    skill = random.choice(current_skills)
                    questions.append(f"Can you explain the concept of {skill}?")
            return questions[:n]

        raise ValueError(f"Expected list of questions, got: {type(questions)}")


# =============================================================================
# QuestionScorer
# =============================================================================

class QuestionScorer:
    """Scores candidate questions on diagnostic value."""

    def score(
        self,
        questions: list[str],
        beliefs: BeliefState,
        history: ConversationHistory
    ) -> list[float]:
        """Score each question 1-10 on diagnostic value."""

        if len(questions) == 1:
            return [5.0]  # Default score for single question

        system_prompt = """You are an expert at evaluating diagnostic questions.
Score each question on how much it would help disambiguate a student's uncertain skills.

Scoring criteria (1-10):
- 10: Perfectly targets uncertain skills, will clearly reveal proficiency level
- 7-9: Good diagnostic value, targets relevant skills
- 4-6: Moderate value, somewhat relevant
- 1-3: Low value, redundant or targets already-known skills

Output ONLY a JSON array of scores, one per question.
Format: [score1, score2, ...]"""

        beliefs_summary = "\n".join(
            f"- {skill}: {beliefs.get_prediction(skill)} ({beliefs.get_confidence(skill):.0%} confident)"
            for skill in get_current_skills()
        )

        questions_list = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(questions)
        )

        user_prompt = f"""Score these {len(questions)} questions on diagnostic value.

Current beliefs about student:
{beliefs_summary}

Conversation history:
{history.format_for_prompt()}

Questions to score:
{questions_list}

Output a JSON array of {len(questions)} scores (1-10):"""

        scores = llm_call_json(system_prompt, user_prompt, temperature=0.3)

        if isinstance(scores, list) and len(scores) >= len(questions):
            return [float(s) for s in scores[:len(questions)]]

        # Fallback: return uniform scores
        return [5.0] * len(questions)


# =============================================================================
# BeliefTracker
# =============================================================================

class BeliefTracker:
    """Updates belief distributions based on Q&A evidence."""

    def update(
        self,
        beliefs: BeliefState,
        history: ConversationHistory,
        n_candidates: int = 1
    ) -> tuple[BeliefState, str]:
        """Update beliefs given the conversation history.

        Number of self-correction passes scales with N:
        - N=1: 1 pass (initial analysis only)
        - N=3: 2 passes (initial + 1 self-correction)
        - N=5+: 3 passes (initial + 2 self-corrections)

        Args:
            beliefs: Current belief state
            history: Conversation history
            n_candidates: Number of candidates (determines thinking passes)

        Returns:
            Tuple of (updated beliefs, full thinking transcript)
        """
        # Determine number of passes based on N
        if n_candidates >= 5:
            num_passes = 3
        elif n_candidates >= 3:
            num_passes = 2
        else:
            num_passes = 1

        all_thinking = []
        current_beliefs = beliefs

        for pass_num in range(num_passes):
            is_first_pass = (pass_num == 0)
            is_final_pass = (pass_num == num_passes - 1)

            # Build pass-specific prompt
            if is_first_pass:
                pass_instruction = "INITIAL ANALYSIS: Analyze what each student answer reveals about their skill levels."
            else:
                pass_instruction = f"SELF-CORRECTION PASS {pass_num}: Review your previous analysis. Did you miss anything? Are any predictions likely WRONG? Correct them."

            # Build dynamic skill list for prompt
            current_skills = get_current_skills()
            skills_list = "\n".join(f"{i+1}. {skill}" for i, skill in enumerate(current_skills))
            prob_format = ", ".join(f'"{s}": [p_low, p_high]' for s in current_skills)

            system_prompt = f"""You are a teacher diagnosing a student's skill levels from their answers.

{pass_instruction}

YOU ARE DIAGNOSING THESE {len(current_skills)} SKILLS:
{skills_list}

EACH SKILL IS EITHER LOW OR HIGH:
- LOW: Student gives WRONG answers, shows confusion, makes fundamental errors
- HIGH: Student answers correctly, explains clearly, shows confidence

HOW TO IDENTIFY SKILL LEVEL FROM ANSWERS:
- Wrong code/explanation about X → X is LOW
- "I don't understand" or "I'm confused" about X → X is LOW
- Correct and confident about X → X is HIGH
- If skill X was NOT tested in any question → keep previous belief

Output format:
<thinking>
[For each skill: was it tested? what did student say? what level does it indicate?]
</thinking>
<probabilities>
{{{prob_format}}}
</probabilities>"""

            beliefs_str = "\n".join(
                f"- {skill}: [low={current_beliefs.distributions[skill][0]:.2f}, "
                f"high={current_beliefs.distributions[skill][1]:.2f}] → prediction: {current_beliefs.get_prediction(skill)}"
                for skill in get_current_skills()
            )

            # Include previous thinking in self-correction passes
            if is_first_pass:
                context = ""
            else:
                context = f"\n\nYOUR PREVIOUS ANALYSIS:\n{all_thinking[-1]}\n\nNow review and correct if needed:"

            user_prompt = f"""Current predictions:
{beliefs_str}

Conversation history:
{history.format_for_prompt()}{context}

Analyze and output updated probabilities:"""

            try:
                response = llm_call(system_prompt, user_prompt, temperature=0.3)

                # Extract thinking
                thinking_text = ""
                if "<thinking>" in response and "</thinking>" in response:
                    start = response.find("<thinking>") + len("<thinking>")
                    end = response.find("</thinking>")
                    thinking_text = response[start:end].strip()

                pass_label = "Initial" if is_first_pass else f"Correction {pass_num}"
                all_thinking.append(f"[{pass_label}]\n{thinking_text}")

                # Extract probabilities
                if "<probabilities>" in response and "</probabilities>" in response:
                    start = response.find("<probabilities>") + len("<probabilities>")
                    end = response.find("</probabilities>")
                    json_str = response[start:end].strip()
                    updated = json.loads(json_str)
                else:
                    updated = parse_json_from_response(response)

                if isinstance(updated, dict):
                    new_distributions = {}
                    for skill in get_current_skills():
                        if skill in updated and isinstance(updated[skill], list):
                            # Binary: take first 2 values
                            probs = [float(p) for p in updated[skill][:2]]
                            # Ensure we have exactly 2 probabilities
                            if len(probs) < 2:
                                probs = probs + [0.5] * (2 - len(probs))
                            total = sum(probs)
                            if total > 0:
                                probs = [p / total for p in probs]
                            else:
                                probs = [0.5, 0.5]
                            new_distributions[skill] = probs
                        else:
                            new_distributions[skill] = current_beliefs.distributions[skill]

                    current_beliefs = BeliefState(distributions=new_distributions)

            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                all_thinking.append(f"[Pass {pass_num + 1}] (parse error)")
                continue

        full_transcript = "\n\n".join(all_thinking)
        return current_beliefs, full_transcript


# =============================================================================
# DiagnosisChecker
# =============================================================================

class DiagnosisChecker:
    """Checks diagnostic completion and accuracy."""

    @staticmethod
    def is_complete(beliefs: BeliefState) -> bool:
        """Check if all skills have reached 90% confidence."""
        return beliefs.all_confident(CONFIDENCE_THRESHOLD)

    @staticmethod
    def check_accuracy(beliefs: BeliefState, ground_truth: SkillProfile) -> dict:
        """Check if predictions match ground truth."""
        results = {}
        for skill in get_current_skills():
            predicted = beliefs.get_prediction(skill)
            actual = ground_truth.levels[skill]
            results[skill] = predicted == actual
        return results

    @staticmethod
    def accuracy_score(beliefs: BeliefState, ground_truth: SkillProfile) -> float:
        """Calculate accuracy as fraction of correct predictions."""
        results = DiagnosisChecker.check_accuracy(beliefs, ground_truth)
        return sum(results.values()) / len(results)

    @staticmethod
    def all_correct(beliefs: BeliefState, ground_truth: SkillProfile) -> bool:
        """Check if ALL predictions match ground truth (100% accuracy)."""
        results = DiagnosisChecker.check_accuracy(beliefs, ground_truth)
        return all(results.values())


# =============================================================================
# Main Diagnostic Loop
# =============================================================================

@dataclass
class GameResult:
    """Results from a single diagnostic game."""
    turns_taken: int
    accuracy: float
    all_correct: bool  # True if all predictions matched ground truth
    ground_truth: SkillProfile
    final_beliefs: BeliefState
    transcript: str = ""  # Full game transcript


def run_game(
    n_candidates: int,
    game_seed: int = 42,
    profile: Optional[SkillProfile] = None,
    profile_idx: Optional[int] = None
) -> GameResult:
    """Run a single diagnostic game.

    Args:
        n_candidates: Number of candidate questions to generate per turn
        game_seed: Random seed for reproducibility (used if profile is None)
        profile: Optional predefined skill profile (if None, generates random)
        profile_idx: Optional profile index for labeling
    """

    # Capture transcript to buffer
    transcript = StringIO()

    def log(msg: str = ""):
        transcript.write(msg + "\n")

    # Initialize components - use provided profile or generate random
    if profile is not None:
        ground_truth = profile
        # Update global SKILLS to match this profile's domain
        game_skills = list(ground_truth.levels.keys())
        set_custom_skills(game_skills)
    else:
        ground_truth = random_profile(seed=game_seed)
        game_skills = get_current_skills()

    student = StudentSimulator(ground_truth)
    generator = QuestionGenerator()
    scorer = QuestionScorer()
    tracker = BeliefTracker()

    beliefs = BeliefState.uniform()
    history = ConversationHistory()

    log(f"{'='*60}")
    if profile is not None:
        label = f"Profile {profile_idx}" if profile_idx is not None else (profile.name or "custom")
        log(f"Game: N={n_candidates}, {label}")
    else:
        log(f"Game: N={n_candidates}, seed={game_seed}")
    log(f"Ground Truth: {ground_truth}")
    log(f"{'='*60}")

    turns_taken = 0

    for turn in range(MAX_TURNS):
        turns_taken = turn + 1
        log(f"\n--- Turn {turns_taken} ---")

        # Generate candidate questions
        candidates = generator.generate(n_candidates, beliefs, history)

        if n_candidates > 1:
            log(f"Generated {len(candidates)} candidates:")
            for i, q in enumerate(candidates, 1):
                log(f"  {i}. {q}")

        # Score and select best question
        if n_candidates > 1:
            scores = scorer.score(candidates, beliefs, history)
            best_idx = scores.index(max(scores))
            best_question = candidates[best_idx]
            log(f"Scores: {[f'{s:.1f}' for s in scores]}")
            log(f"Selected: #{best_idx + 1}")
        else:
            best_question = candidates[0]

        log(f"\nQ: {best_question}")

        # Get student answer - pass history for consistency
        answer = student.answer(best_question, history=history)
        history.add(best_question, answer)
        log(f"\nA: {answer}")

        # Update beliefs with thinking budget based on N
        beliefs, thinking = tracker.update(beliefs, history, n_candidates=n_candidates)

        # Log thinking if present
        if thinking:
            log(f"\n[Thinking - N={n_candidates}]:")
            log(thinking)

        log(f"\nUpdated beliefs:")
        log(str(beliefs))

        # Check correctness feedback - show which predictions are right/wrong
        accuracy_check = DiagnosisChecker.check_accuracy(beliefs, ground_truth)
        correct_count = sum(accuracy_check.values())
        num_skills = len(get_current_skills())
        log(f"\nCorrectness check ({correct_count}/{num_skills} correct):")
        for skill in get_current_skills():
            pred = beliefs.get_prediction(skill)
            is_correct = accuracy_check[skill]
            mark = "✓" if is_correct else "✗"
            log(f"  {skill}: {pred} {mark}")

        # Check completion - END WHEN ALL PREDICTIONS ARE CORRECT
        if DiagnosisChecker.all_correct(beliefs, ground_truth):
            log(f"\n*** ALL PREDICTIONS CORRECT! Diagnosis complete. ***")
            break

    # Calculate final accuracy
    accuracy = DiagnosisChecker.accuracy_score(beliefs, ground_truth)
    is_all_correct = DiagnosisChecker.all_correct(beliefs, ground_truth)

    log(f"\n{'='*60}")
    log(f"GAME COMPLETE")
    log(f"{'='*60}")
    log(f"Turns taken: {turns_taken}")
    log(f"All correct: {is_all_correct}")
    num_skills = len(get_current_skills())
    log(f"Accuracy: {accuracy:.0%} ({int(accuracy * num_skills)}/{num_skills} skills correct)")
    log(f"")
    log(f"Ground Truth:  {ground_truth}")
    log(f"Predictions:   {', '.join(f'{s}: {beliefs.get_prediction(s)}' for s in get_current_skills())}")
    log(f"")

    # Show per-skill results
    for skill in get_current_skills():
        pred = beliefs.get_prediction(skill)
        actual = ground_truth.levels[skill]
        match = "✓" if pred == actual else "✗"
        conf = beliefs.get_confidence(skill)
        log(f"  {skill}: predicted {pred} (conf {conf:.0%}), actual {actual} {match}")

    log(f"{'='*60}")

    return GameResult(
        turns_taken=turns_taken,
        accuracy=accuracy,
        all_correct=is_all_correct,
        ground_truth=ground_truth,
        final_beliefs=beliefs,
        transcript=transcript.getvalue()
    )


# =============================================================================
# Experiment Runner
# =============================================================================

def create_output_dir() -> str:
    """Create timestamped output directory for game results."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("games", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_experiment(
    n_values: list[int],
    num_profiles: int,
    base_seed: int = 42,
    parallel: bool = True
) -> dict:
    """Run the full experiment.

    IMPORTANT: Each profile is tested with ALL N values for fair comparison.
    N values are run IN PARALLEL for each profile for faster execution.

    Results are saved to games/<timestamp>/ folder with separate files per N value.
    """

    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")

    # Pre-generate all profiles for reproducibility
    profiles = []
    for i in range(num_profiles):
        profiles.append(random_profile(seed=base_seed + i))

    print(f"Generated {len(profiles)} student profiles.")
    print(f"Testing each profile with N values: {n_values} {'(parallel)' if parallel else '(sequential)'}")
    print(f"{'='*60}\n")

    # Results: n_value -> list of game results
    results_by_n = {n: [] for n in n_values}

    # Transcripts: n_value -> list of transcripts
    transcripts_by_n = {n: [] for n in n_values}

    def run_single_game(n: int, profile: SkillProfile, profile_idx: int):
        """Run a single game - used for parallel execution."""
        result = run_game(
            n_candidates=n,
            profile=profile,
            profile_idx=profile_idx
        )
        return n, result

    for profile_idx, profile in enumerate(profiles):
        domain_label = f"[{profile.domain}]" if profile.domain else ""
        print(f"Profile {profile_idx + 1}/{len(profiles)} {domain_label}: {profile}")
        print(f"  Running N={n_values}...", end=" ", flush=True)

        if parallel:
            # Run all N values in parallel
            with ThreadPoolExecutor(max_workers=len(n_values)) as executor:
                futures = {
                    executor.submit(run_single_game, n, profile, profile_idx + 1): n
                    for n in n_values
                }

                completed_results = {}
                for future in as_completed(futures):
                    n = futures[future]
                    try:
                        n, result = future.result()
                        completed_results[n] = result
                    except Exception as e:
                        print(f"N={n} Error: {e}", end="  ")

            # Store results in order
            for n in n_values:
                if n in completed_results:
                    result = completed_results[n]
                    results_by_n[n].append(result)
                    transcripts_by_n[n].append(result.transcript)

            # Print summary
            parts = []
            for n in n_values:
                if n in completed_results:
                    r = completed_results[n]
                    parts.append(f"N={n}:t={r.turns_taken},acc={r.accuracy:.0%}")
            print(" | ".join(parts))

        else:
            # Sequential execution
            for n in n_values:
                try:
                    result = run_game(
                        n_candidates=n,
                        profile=profile,
                        profile_idx=profile_idx + 1
                    )
                    results_by_n[n].append(result)
                    transcripts_by_n[n].append(result.transcript)
                    print(f"N={n}:t={result.turns_taken},acc={result.accuracy:.0%}", end="  ")
                except Exception as e:
                    print(f"N={n} Error: {e}", end="  ")
            print()

    # Write transcripts to files
    print(f"\nWriting transcripts to {output_dir}/")
    for n in n_values:
        filename = os.path.join(output_dir, f"n_{n}.txt")
        with open(filename, "w") as f:
            f.write(f"EXPERIMENT RESULTS: N={n}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Number of profiles tested: {len(transcripts_by_n[n])}\n")
            f.write(f"N values tested: {n_values}\n")
            f.write(f"{'='*60}\n\n")

            for i, transcript in enumerate(transcripts_by_n[n], 1):
                f.write(f"\n{'#'*60}\n")
                f.write(f"# GAME {i}\n")
                f.write(f"{'#'*60}\n\n")
                f.write(transcript)
                f.write("\n")

        print(f"  Written: {filename}")

    # Write summary file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Profiles tested: {num_profiles}\n")
        f.write(f"N values: {n_values}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"{'N':>6} | {'Avg Turns':>10} | {'Accuracy':>10} | {'Success':>10}\n")
        f.write("-" * 50 + "\n")

        for n in n_values:
            game_results = results_by_n[n]
            if game_results:
                avg_turns = sum(r.turns_taken for r in game_results) / len(game_results)
                avg_accuracy = sum(r.accuracy for r in game_results) / len(game_results)
                success_rate = sum(r.all_correct for r in game_results) / len(game_results)
                f.write(f"{n:>6} | {avg_turns:>10.2f} | {avg_accuracy:>9.1%} | {success_rate:>9.1%}\n")

    print(f"  Written: {summary_file}")

    # Aggregate results for return
    results = {}
    for n, game_results in results_by_n.items():
        if game_results:
            avg_turns = sum(r.turns_taken for r in game_results) / len(game_results)
            avg_accuracy = sum(r.accuracy for r in game_results) / len(game_results)
            success_rate = sum(r.all_correct for r in game_results) / len(game_results)

            results[n] = {
                "avg_turns": avg_turns,
                "avg_accuracy": avg_accuracy,
                "success_rate": success_rate,
                "games": len(game_results)
            }

    return results


def print_results_table(results: dict):
    """Print a formatted results table."""

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"{'N':>6} | {'Avg Turns':>10} | {'Accuracy':>10} | {'Success':>10} | {'Games':>6}")
    print("-" * 70)

    for n, data in sorted(results.items()):
        print(
            f"{n:>6} | "
            f"{data['avg_turns']:>10.2f} | "
            f"{data['avg_accuracy']:>9.1%} | "
            f"{data['success_rate']:>9.1%} | "
            f"{data['games']:>6}"
        )

    print("=" * 70)
    print("\nKey:")
    print("  N: Number of candidate questions generated per turn")
    print("  Avg Turns: Average turns to correctly diagnose all skills (max 20)")
    print("  Accuracy: Average fraction of skills correctly identified")
    print("  Success: Fraction of games where all 5 skills were correctly diagnosed")


# =============================================================================
# Demo Mode - Run with Sample Profiles
# =============================================================================

def run_demo(profile_name: str, n_candidates: int = 10) -> None:
    """Run a single demo game with a sample profile."""
    print("=" * 70)
    print("DEMO: Diagnosing a Sample Student Profile")
    print("=" * 70)

    profile = get_sample_profile(profile_name)
    print(f"\nUsing profile: {profile_name}")
    print(f"N candidates: {n_candidates}")
    print(f"Ground Truth: {profile}")
    print()

    result = run_game(
        n_candidates=n_candidates,
        profile=profile
    )

    # Print the transcript
    print(result.transcript)

    print(f"\n{'='*70}")
    print("DEMO SUMMARY")
    print(f"{'='*70}")
    print(f"Profile: {profile_name}")
    print(f"Ground Truth: {profile}")
    print(f"Turns taken: {result.turns_taken}")
    print(f"Accuracy: {result.accuracy:.0%}")
    print(f"All skills correct: {result.all_correct}")


def run_all_sample_profiles(n_candidates: int = 10) -> None:
    """Run diagnosis on all sample profiles and summarize results."""
    print("=" * 70)
    print("Running Diagnosis on All Sample Profiles")
    print("=" * 70)
    print(f"N candidates per turn: {n_candidates}")

    results = []
    for name, profile in SAMPLE_PROFILES.items():
        print(f"  {name}...", end=" ", flush=True)
        result = run_game(
            n_candidates=n_candidates,
            profile=profile
        )
        results.append((name, result))
        print(f"turns={result.turns_taken}, acc={result.accuracy:.0%}")

    print(f"\n{'='*70}")
    print("SUMMARY: All Sample Profiles")
    print(f"{'='*70}")
    print(f"{'Profile':<30} | {'Turns':>6} | {'Accuracy':>8} | {'Success':>9}")
    print("-" * 70)
    for name, result in results:
        print(f"{name:<30} | {result.turns_taken:>6} | {result.accuracy:>7.0%} | "
              f"{'Yes' if result.all_correct else 'No':>9}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the full experiment or demo based on command line args."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inverse Cognitive Search: Test-Time Compute for Student Diagnosis"
    )
    parser.add_argument(
        "--demo",
        type=str,
        help="Run a demo with a specific sample profile"
    )
    parser.add_argument(
        "--demo-all",
        action="store_true",
        help="Run demos on all sample profiles"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all available sample profiles"
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List all available skill domains"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=list(SKILL_DOMAINS.keys()),
        help="Set the skill domain (e.g., math, physics, chemistry)"
    )
    parser.add_argument(
        "--skills",
        type=str,
        help="Set custom skills as comma-separated list (e.g., 'fractions,decimals,percentages')"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of candidate questions per turn (default: 10)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=30,
        help="Number of games per N value in experiment (default: 30)"
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="1,5,10,25",
        help="Comma-separated N values to test (default: 1,5,10,25)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run N values sequentially instead of in parallel"
    )

    args = parser.parse_args()

    # Handle domain/skills configuration
    if args.skills:
        # Custom skills override domain
        skills = [s.strip() for s in args.skills.split(",")]
        set_custom_skills(skills)
        print(f"Using custom skills: {skills}")
    elif args.domain:
        set_skill_domain(args.domain)
        print(f"Using domain: {args.domain}")

    # Get current skills (may have been updated)
    current_skills = get_current_skills()

    # Handle special modes
    if args.list_domains:
        print("\nAvailable Skill Domains:")
        print("=" * 60)
        for domain, skills in SKILL_DOMAINS.items():
            print(f"  {domain}: {', '.join(skills)}")
        print("\nUse --domain <name> to select a domain")
        print("Use --skills 'skill1,skill2,skill3' for custom skills")
        return

    if args.list_profiles:
        list_sample_profiles()
        return

    if args.demo:
        run_demo(args.demo, n_candidates=args.n)
        return

    if args.demo_all:
        run_all_sample_profiles(n_candidates=args.n)
        return

    # Default: run full experiment
    print("=" * 70)
    print("Inverse Cognitive Search: Test-Time Compute for Student Diagnosis")
    print("=" * 70)
    if args.domain or args.skills:
        print(f"Skills: {', '.join(current_skills)}")
    else:
        print(f"Domains: {', '.join(SKILL_DOMAINS.keys())} (random)")
    print(f"Levels: {', '.join(LEVELS)}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}, Max turns: {MAX_TURNS}")

    # Parse N values
    n_values = [int(x.strip()) for x in args.n_values.split(",")]
    num_profiles = args.games

    # Run experiment with time-based seed for diversity
    import time
    base_seed = int(time.time()) % 10000  # Use current time for variety
    results = run_experiment(
        n_values=n_values,
        num_profiles=num_profiles,
        base_seed=base_seed,
        parallel=not args.sequential
    )

    # Print results
    print_results_table(results)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
