import argparse
import asyncio
import random
from dataclasses import dataclass
from typing import List
from .student import SimulatedStudent, StudentProfile
from .teacher import SimulatedTeacher
from .judge import Judge
from .logger import RunLogger, RunLog, TurnLog
from .profiles import PROFILES, get_profiles_by_dimensions

DEFAULT_MAX_TURNS = 3

DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
]

DEFAULT_PROFILES = ["programming_novice", "calculus_foundations", "algebra_beginner"]


@dataclass
class EvalResult:
    model: str
    profile: str
    n: int
    accuracy: float
    turns_taken: int
    perfect: bool
    log: RunLog


async def run_evaluation(model: str, profile: StudentProfile, profile_name: str, max_turns: int, n: int, logger: RunLogger) -> EvalResult:
    student = SimulatedStudent(profile)
    teacher = SimulatedTeacher(
        subject=profile.subject,
        topics=list(profile.knowledge_state.keys()),
        model=model,
    )
    judge = Judge(profile)
    turn_logs: List[TurnLog] = []

    for turn in range(1, max_turns + 1):
        q_result = await teacher.agenerate_question(n=n)
        answer = await student.aanswer(q_result.question)
        teacher.record_turn(q_result.question, answer.final)

        guess = await teacher.aassess()
        result = await judge.agrade(guess)

        turn_logs.append(TurnLog(
            turn=turn,
            question=q_result.question,
            answer=answer.final,
            guess={e.topic: {"known": e.known, "confidence": e.confidence, "reasoning": e.reasoning} for e in guess.estimates},
            accuracy=result.accuracy,
            candidates=q_result.candidates,
            selected_idx=q_result.selected_idx,
            initial_answer=answer.initial,
            reflection=answer.reflection,
            topics_tested=answer.classification.topics_tested if answer.classification else None,
            should_know=answer.classification.should_know if answer.classification else None,
            classification_reasoning=answer.classification.reasoning if answer.classification else None,
        ))

        if result.accuracy == 1.0:
            log = RunLog(
                model=model, profile=profile_name, subject=profile.subject,
                ground_truth=profile.knowledge_state, turns=turn_logs,
                final_accuracy=1.0, perfect=True, n=n,
            )
            logger.save(log)
            return EvalResult(model=model, profile=profile_name, n=n, accuracy=1.0, turns_taken=turn, perfect=True, log=log)

    log = RunLog(
        model=model, profile=profile_name, subject=profile.subject,
        ground_truth=profile.knowledge_state, turns=turn_logs,
        final_accuracy=result.accuracy, perfect=False, n=n,
    )
    logger.save(log)
    return EvalResult(model=model, profile=profile_name, n=n, accuracy=result.accuracy, turns_taken=max_turns, perfect=False, log=log)


async def run_all(models: List[str], profile_names: List[str], max_turns: int, ns: List[int], logger: RunLogger) -> List[EvalResult]:
    tasks = []
    for model in models:
        for name in profile_names:
            for n in ns:
                profile = PROFILES[name]
                tasks.append(run_evaluation(model, profile, name, max_turns, n, logger))

    return await asyncio.gather(*tasks)


def resolve_profiles(profile_args: List[str]) -> List[str]:
    """Resolve profile arguments - can be names or dimension numbers (1-10)."""
    resolved = []

    for p in profile_args:
        # Check if it's a dimension number
        if p.isdigit():
            dim = int(p)
            dim_profiles = get_profiles_by_dimensions(dim)
            if not dim_profiles:
                print(f"Error: No profiles with {dim} dimensions")
                return []
            # Pick one random profile from this dimension
            name = random.choice(list(dim_profiles.keys()))
            resolved.append(name)
        elif p in PROFILES:
            resolved.append(p)
        else:
            print(f"Error: Unknown profile '{p}'")
            print(f"Use profile names or dimension numbers (1-10)")
            return []

    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES,
                        help="Profile names or dimension numbers (1-10)")
    parser.add_argument("--n", nargs="+", type=int, default=[1], help="Best-of-N question generation")
    args = parser.parse_args()

    # Resolve profile names (supports dimension numbers)
    profile_names = resolve_profiles(args.profiles)
    if not profile_names:
        return

    logger = RunLogger()
    run_dir = logger.start_run()

    print(f"Max {args.max_turns} turns | N={args.n}", f"Models: {', '.join(args.models)}", f"Profiles: {', '.join(profile_names)}", f"Logs: {run_dir}", sep="\n")
    results = asyncio.run(run_all(args.models, profile_names, args.max_turns, args.n, logger))

    print("\n" + "=" * 60, "\nRESULTS", "\n" + "=" * 60)

    for model in args.models:
        print(f"\n{model}", "-" * 40, sep="\n")
        for n in args.n:
            print(f"  n={n}:")
            model_n_results = [r for r in results if r.model == model and r.n == n]
            for r in model_n_results:
                print(f"    [{r.profile}] {r.accuracy:.0%} in {r.turns_taken} turns ({'perfect' if r.perfect else 'incomplete'})")


if __name__ == "__main__":
    main()
