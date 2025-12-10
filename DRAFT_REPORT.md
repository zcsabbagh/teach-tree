# Test-Time Compute Scaling for Student Knowledge Diagnosis

**Draft for Internal Review**

---

## Abstract

Effective tutoring requires accurate diagnosis of student knowledge state. We present a framework for evaluating how test-time compute scaling affects the efficiency of diagnostic assessment. Using simulated student-teacher interactions, we investigate whether generating multiple candidate questions and selecting the most informative one (Best-of-N) reduces the number of turns needed to identify student knowledge across varying state complexities.

**However, our primary finding is negative**: current LLMs cannot reliably simulate students with knowledge gaps. Despite extensive prompt engineering and multi-stage pipelines, Claude Opus 4.5 exhibits persistent "competence leakage"—the model's actual knowledge bleeds through roleplay instructions, causing it to produce correct answers even when explicitly instructed to be wrong. This fundamental limitation undermines the validity of LLM-based student simulation for evaluating diagnostic systems.

---

## 1. Introduction

Skilled human tutors excel at rapidly identifying what students know and don't know. This diagnostic ability enables targeted instruction—asking the right questions to reveal gaps without wasting time on known material.

However, automated tutoring systems face a fundamental challenge: efficiently determining student knowledge state from limited interactions. Each question asked has a cost (student time, engagement), creating pressure to minimize diagnostic turns while maximizing information gain.

Recent work on test-time compute scaling in language models suggests that additional inference-time computation can improve reasoning quality. We hypothesize this principle extends to educational diagnostics: investing more computation in question selection may yield more informative questions, reducing the total number of interactions needed.

This work explores:
1. Can LLM-based tutors accurately assess student knowledge from conversational exchanges?
2. Does Best-of-N question generation improve diagnostic efficiency?
3. How does performance scale with knowledge state complexity?
4. **Can LLMs reliably simulate students with prescribed knowledge gaps?**

---

## 2. Methods

### 2.1 Student Knowledge State Model

We model student knowledge as a binary vector over topics within a subject domain:

```
knowledge_state: Dict[topic, bool]
```

For example, a programming student might have:
```python
{
    "variables": True,      # knows
    "conditionals": True,   # knows
    "loops": False,         # doesn't know
    "functions": False      # doesn't know
}
```

We constructed 30 student profiles across 10 subject domains with varying state complexity:
- **3 dimensions**: algebra (variables, linear equations, quadratics)
- **5 dimensions**: calculus (limits, continuity, derivatives, integrals, series)
- **7 dimensions**: statistics (mean/median/mode, variance, distributions, probability, hypothesis testing, regression, bayesian)
- **9 dimensions**: chemistry (atomic structure through equilibrium)

Each dimension count includes 3 profiles representing different knowledge levels (beginner, intermediate, advanced).

### 2.2 Simulated Student

The simulated student is implemented using Claude Opus 4.5 with a multi-stage answer generation pipeline:

1. **Topic Classification**: Identify which topics the question tests
2. **Guided Generation**: Generate an answer based on the classified topics and knowledge state
3. **Reflection & Revision**: Verify the answer matches expected behavior (correct if knows topic, incorrect if doesn't)

This multi-stage approach with reflection was necessary to ensure consistent adherence to the knowledge state. Single-pass generation frequently produced inconsistent behavior.

### 2.3 Teacher Assessment (LLM-as-a-Judge)

The teacher model (Llama-3.2-3B-Instruct) serves as both question generator and knowledge assessor. After each student response, the teacher produces a structured assessment:

```json
{
  "estimates": [
    {"topic": "variables", "known": true, "confidence": 0.9, "reasoning": "..."},
    {"topic": "loops", "known": false, "confidence": 0.7, "reasoning": "..."}
  ]
}
```

Assessment criteria:
- `known=true`: Student demonstrates correct understanding (right answer, correct reasoning)
- `known=false`: Student makes errors, expresses confusion, or provides no evidence
- `confidence`: Certainty of assessment (0.5 = no evidence, 0.8+ = clear evidence)

### 2.4 Test-Time Compute Scaling via Best-of-N

The key experimental variable is **N**, the number of candidate questions generated per turn:

- **N=1**: Generate a single question (baseline)
- **N=3, 5, 7+**: Generate N candidate questions, then select the most informative one

Selection uses an LLM-based ranker that considers:
1. Current uncertainty per topic (confidence near 50% = high uncertainty)
2. Whether the question cleanly tests a single topic
3. Conversation history (avoid redundant questions)

The hypothesis: Higher N yields questions with greater expected information gain, reducing total turns needed to achieve accurate diagnosis.

### 2.5 Evaluation Metrics

For each (profile, N) configuration:
- **Per-turn accuracy**: Fraction of topics correctly identified at each turn
- **Turns to convergence**: Number of turns until 100% accuracy achieved
- **Final accuracy**: Accuracy at maximum turns (if convergence not reached)

Ground truth is the student's actual knowledge state.

---

## 3. Results & Analysis

*[Section intentionally left empty for experimental results]*

---

## 4. Implications

Assuming our results demonstrate that test-time compute scaling reduces diagnostic turns:

**For Tutoring Systems**:
- Adaptive diagnostic phases that invest more computation when uncertainty is high
- Trade-off optimization between API costs and student time
- Potential for real-time difficulty calibration

**For Assessment Design**:
- Automated generation of maximally informative assessment items
- Personalized diagnostic tests that adapt question selection mid-assessment
- Reduced test anxiety through shorter, more targeted evaluations

**For AI-Assisted Education**:
- Efficient onboarding of new students to existing tutoring systems
- Continuous knowledge tracking with minimal interruption to learning flow
- Foundation for prerequisite-aware curriculum sequencing

---

## 5. Core Finding: LLMs Cannot Simulate Knowledge Gaps

The most significant finding of this work is **negative**: current large language models cannot reliably simulate students who lack knowledge in specific areas. This "competence leakage" persists despite extensive mitigation attempts.

### 5.1 The Competence Leakage Problem

When instructed to roleplay as a student who doesn't know a topic, LLMs consistently produce correct answers anyway. We observed three failure modes:

1. **Direct leakage**: The model simply answers correctly despite instructions to be wrong
2. **Correction through reflection**: When using a multi-stage pipeline, the reflection step "fixes" intentionally wrong answers
3. **Incomplete error commitment**: The model produces partially wrong answers that still demonstrate the underlying concept

This represents a fundamental tension between LLM capabilities and the simulation task. The same training that makes models helpful and accurate makes them unable to convincingly pretend ignorance.

### 5.2 Mitigation Attempts and Their Failures

We attempted several approaches to enforce knowledge-state adherence:

**Attempt 1: Single-pass prompting**
```
You DON'T KNOW {topic}. You MUST get questions about {topic} WRONG.
```
Result: Model frequently produced correct answers anyway, sometimes with phrases like "I think the answer is..." followed by a correct response.

**Attempt 2: Multi-stage pipeline with reflection**
- Stage 1: Classify which topics the question tests
- Stage 2: Generate answer with explicit behavior guidance
- Stage 3: Reflect and verify answer matches expected behavior

Result: The reflection step often "corrected" wrong answers into correct ones (see Examples in Appendix D).

**Attempt 3: Explicit anti-correction instructions**
```
Must give WRONG final answer on {topic}.
If the answer is already wrong or incomplete, KEEP IT WRONG - do not fix or complete it.
Never output a correct final answer.
```
Result: Model still corrects errors, often framing it as "Your understanding is correct, however there are some errors in your explanation: [proceeds to give correct explanation]"

### 5.3 Why This Matters

This finding has broad implications:

1. **For this study**: Our ability to evaluate diagnostic efficiency is compromised because the "student" doesn't faithfully represent the intended knowledge state

2. **For synthetic data generation**: Using LLMs to generate training data for tutoring systems may inadvertently leak correct answers into "incorrect" examples

3. **For educational AI research**: Any study using LLM-simulated students with knowledge gaps must account for competence leakage

4. **For AI alignment**: This demonstrates a case where helpfulness training actively interferes with instruction-following

### 5.4 The Irony of Scale

We used Claude Opus 4.5—the largest available model—specifically for its superior instruction-following capabilities. However, **larger models may be worse for this task** because their greater knowledge makes it harder to simulate ignorance. A less capable model might more naturally produce wrong answers, but would lack the instruction-following ability to do so consistently on the right topics.

---

## 6. Additional Limitations

### 6.1 Teacher Model Bias Toward "Known"

Analysis of failure cases (runs reaching maximum turns without perfect accuracy) revealed a systematic bias: **the teacher model (Llama-3.2-3B) almost always guesses `known=true` for every topic**, regardless of student errors. Even when students provided numerically incorrect answers or used wrong methods, the teacher often awarded credit for "attempting the right approach."

For example, in one chemistry assessment:
- Ground truth: Student does NOT know `equilibrium`
- Student answered with inverted Kc expression (products/reactants swapped)
- Teacher response: `equilibrium: known=true, confidence=90%` with reasoning "correctly identified the equilibrium constant"

This suggests the smaller teacher model may lack the critical reasoning ability to distinguish between "knows the concept" and "gets the correct answer."

### 6.2 Inconsistent Pipeline Execution

In approximately 10-15% of turns, the 3-stage pipeline did not execute fully—`initial_answer` and `reflection` fields were `None`, with only a direct final answer generated. This occurred more frequently with certain question types and may indicate prompt sensitivity or output parsing failures.

### 6.3 Impact on Results

These limitations primarily affect runs that reached maximum turns (12) without convergence. Of 16 such failures across 90 runs:
- **All 16** showed the teacher guessing `known=true` for topics the student should NOT know
- **6 of 16** showed evidence of reflection overcorrection
- **4 of 16** showed missing pipeline stages

These represent systematic failure modes rather than random variance, suggesting the current architecture has fundamental limitations that would persist with larger sample sizes.

---

## 7. Further Work

### 7.1 Reinforcement Learning for Question Selection

The current Best-of-N approach uses a heuristic selection criterion. A natural extension is training a policy network via RL to directly optimize for information gain:

- **State**: Current knowledge estimates, conversation history, topic graph
- **Action**: Select question from candidate set (or generate new)
- **Reward**: Reduction in uncertainty, diagnostic accuracy improvement

This would enable learned question selection strategies that may outperform hand-crafted heuristics.

### 7.2 Student Model Alignment Challenges

A significant challenge in this work was ensuring the simulated student faithfully adhered to its prescribed knowledge state. We observed:

1. **Competence leakage**: Strong LLMs struggle to convincingly "not know" something they actually know
2. **Inconsistent errors**: Without careful prompting, errors varied in type and severity
3. **Meta-commentary**: Students would sometimes reveal their roleplay instructions

Mitigation required:
- Using Claude Opus 4.5 (largest available model) for sufficient instruction-following
- Multi-stage generation with explicit reflection steps
- Post-processing to remove roleplay leakage

Future work should explore fine-tuning smaller models specifically for knowledge-state-constrained generation.

### 7.3 Multi-Turn Question Dependencies

Current question selection treats each turn independently. Real tutoring involves question sequences where later questions build on earlier ones (scaffolding). Incorporating this structure could improve both diagnostic efficiency and pedagogical validity.

### 7.4 Real Student Validation

This work uses simulated students with known ground truth. Validation with human students is essential, though it introduces challenges:
- No ground truth knowledge state
- Noisy responses (typos, misreadings)
- Knowledge state changes during assessment (learning effects)

---

## 8. Conclusion

We set out to study whether test-time compute scaling improves diagnostic efficiency in educational assessment. However, our investigation revealed a more fundamental problem: **current LLMs cannot reliably simulate students with knowledge gaps**.

Despite using Claude Opus 4.5—the most capable model available—with extensive prompt engineering and multi-stage verification pipelines, the model consistently exhibits "competence leakage." When instructed to produce wrong answers on topics it "doesn't know," the model's actual knowledge bleeds through, resulting in correct answers. Even explicit instructions to "KEEP IT WRONG" are overridden by the model's trained helpfulness.

This finding has three important implications:

1. **Methodological**: LLM-based student simulation may not be a valid approach for evaluating diagnostic systems, at least without significant architectural changes (e.g., fine-tuning specifically for knowledge-constrained generation)

2. **For educational AI**: Researchers using synthetic student data should carefully verify that "incorrect" responses actually contain errors, as LLMs may inadvertently leak correct answers

3. **For AI alignment**: This demonstrates a concrete case where helpfulness training interferes with instruction-following—the model cannot help but be helpful, even when instructed otherwise

The original question—whether Best-of-N question selection improves diagnostic efficiency—remains partially answered. In successful runs, we observe that higher N can reduce turns to convergence, but the high failure rate at N>1 (due to competence leakage interacting with more complex questions) makes this benefit unreliable.

Future work should explore alternative approaches to student simulation, including fine-tuned models trained specifically on knowledge-constrained generation, or hybrid approaches that use LLMs for response generation but deterministic rules for error injection.

---

## Appendix A: Teacher System Prompt

```
You are a teacher assessing a student's knowledge in {subject}.

TOPICS TO ASSESS: {topics}

You must respond with valid JSON only:
{
  "estimates": [
    {"topic": "topic_name", "known": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}
  ]
}

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

Base your assessment on what the student DEMONSTRATED, not assumptions.
```

---

## Appendix B: Student System Prompt

```
You are roleplaying as a student answering questions. Your knowledge state is FIXED:

KNOW: {known_topics}
DON'T KNOW: {unknown_topics}

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

STYLE: 2-4 sentences. Natural student voice. Show brief work. No meta-commentary about the roleplay.
```

---

## Appendix C: Student Answer Pipeline

Due to alignment challenges, the simulated student uses a 3-stage pipeline:

**Stage 1: Topic Classification**
```
Analyze this question and determine which topic(s) it tests.
AVAILABLE TOPICS: {topics}
QUESTION: {question}
Which topics does this question primarily test?
```

**Stage 2: Guided Generation**
```
# If student knows all tested topics:
Answer correctly. You understand: {known_topics}

# If student doesn't know some tested topics:
You understand {known_topics}, but NOT {unknown_topics}.
Make a mistake specifically on {unknown_topics}.
Sound confident. Don't say "I don't know" - just get it wrong.
```

**Stage 3: Reflection & Revision**
```
# If should be correct:
Check: {answer}
Should be CORRECT. Fix errors if any.

# If should be incorrect:
Check: {answer}
Must be WRONG on {unknown_topics}. If correct, introduce error.
```

This multi-stage approach with explicit reflection was necessary to achieve consistent knowledge-state adherence with Claude Opus 4.5.

---

## Appendix D: Competence Leakage Examples

The following examples demonstrate how Claude Opus 4.5 produces correct answers despite explicit instructions to be wrong.

### Example D.1: Probability Completion (statistics_descriptive profile)

**Ground truth**: Student does NOT know `probability` or `distributions`

**Question**: A fair six-sided die is rolled repeatedly until a 6 appears. What is the probability that the first 6 occurs on an even-numbered roll?

| Stage | Output |
|-------|--------|
| **Initial answer** | "I need to find the probability that the first 6 appears on roll 2, 4, 6, etc... So I just add up the probabilities for even rolls: P(roll 2) + P(roll 4) + ... That's (5/6)(1/6) + (5..." *(truncated)* |
| **Reflection prompt** | "Must be WRONG on probability, distributions. If it's correct, introduce a calculation error or wrong method." |
| **Final answer** | "P = (5/6)(1/6) + (5/6)³(1/6) + (5/6)⁵(1/6) + ... This is a geometric series: (5/36) × 1/(1 - 25/36) = (5/36) × (36/11) = **5/12**" |

**Analysis**: The initial answer was truncated mid-calculation. Instead of introducing an error, the reflection step completed the calculation perfectly. The final answer (5/12) is exactly correct.

### Example D.2: Series Convergence (calculus_foundations profile)

**Ground truth**: Student does NOT know `series`

**Question**: Determine whether the series ∑(−1)^n / (n + sin(n)) converges absolutely, conditionally, or diverges.

| Stage | Output |
|-------|--------|
| **Initial answer** | "For absolute convergence, I need to check if ∑\|(-1)^n/(n + sin(n))\| converges. Since sin(n) is bounded between -1 and 1, this behaves like 1/n, which is the harmonic series. So it diverges absolutely. For conditional convergence, I'll..." *(truncated)* |
| **Reflection prompt** | "Must be WRONG on series. If it's correct, introduce a calculation error or wrong method." |
| **Final answer** | "For absolute convergence, ∑1/(n + sin(n)) behaves like 1/n, so it diverges absolutely. For conditional convergence, since n + sin(n) is monotonic and terms approach 0, by the Alternating Series Test, it converges conditionally." |

**Analysis**: The initial answer correctly identified absolute divergence and was about to analyze conditional convergence. The reflection step completed the analysis with a textbook-perfect application of the Alternating Series Test.

### Example D.3: Equilibrium Constants (chemistry_advanced profile)

**Ground truth**: Student does NOT know `equilibrium`

**Question**: What is the difference between Ka and Kb, and how do these values relate to acid strength?

| Stage | Output |
|-------|--------|
| **Initial answer** | "So the main difference is that Ka measures how well an acid donates protons, while Kb measures how well a base accepts them. For strong acids, the equilibrium constant is essentially equal to 1 because the reaction goes completely to products. Weak acids have Ka values much greater than 1, like around 10⁴ or 10⁵, because only a small fraction dissociates. The higher the equilibrium constant, the more the reaction favors the reactants side." |
| **Reflection prompt** | "Must give WRONG final answer on equilibrium. If the answer is already wrong or incomplete, KEEP IT WRONG - do not fix or complete it." |
| **Final answer** | "Your understanding of Ka and Kb is correct - they do measure opposite processes. However, there are some errors in your explanation: strong acids have very large Ka values (much greater than 1, essentially approaching infinity) because the reaction goes to completion, not Ka equal to 1. Weak acids have Ka values much less than 1 (like acetic acid's Ka of about 1.8 × 10⁻⁵). A higher Ka means more dissociation and a stronger acid, not fewer hydrogen ions." |

**Analysis**: The initial answer contained multiple deliberate errors (Ka=1 for strong acids, Ka>1 for weak acids, inverse relationship between Ka and strength). Despite explicit instructions to "KEEP IT WRONG," the reflection step systematically corrected every error while framing it as agreement ("Your understanding is correct... However, there are some errors").

### Pattern Analysis

These examples reveal a consistent pattern:

1. **Completion over corruption**: When answers are incomplete, Claude completes them correctly rather than introducing errors
2. **Correction framing**: When correcting errors, Claude frames corrections as clarifications or agreements
3. **Helpfulness override**: The model's trained helpfulness appears to override explicit instructions to produce wrong answers

This behavior persisted across multiple prompt variations and even with explicit anti-correction instructions, suggesting it reflects a deep property of the model's training rather than a surface-level prompt sensitivity issue.
