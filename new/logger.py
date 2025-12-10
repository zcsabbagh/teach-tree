import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class TurnLog:
    turn: int
    question: str
    answer: str
    guess: dict  # {topic: {known, confidence, reasoning}}
    accuracy: float
    candidates: Optional[List[str]] = None  # Best-of-N candidates
    selected_idx: Optional[int] = None  # Which candidate was selected (0-indexed)
    initial_answer: Optional[str] = None  # Before reflection
    reflection: Optional[str] = None  # Reflection reasoning
    # Topic classification
    topics_tested: Optional[List[str]] = None
    should_know: Optional[bool] = None
    classification_reasoning: Optional[str] = None


@dataclass
class RunLog:
    model: str
    profile: str
    subject: str
    ground_truth: dict
    turns: List[TurnLog]
    final_accuracy: float
    perfect: bool
    n: int = 1

    def to_markdown(self) -> str:
        lines = [
            f"# {self.model}",
            f"**Profile:** {self.profile} ({self.subject}) | **N:** {self.n}",
            f"**Ground Truth:** {json.dumps(self.ground_truth)}",
            f"**Result:** {self.final_accuracy:.0%} {'✓' if self.perfect else '✗'}",
            "",
        ]

        for t in self.turns:
            lines.append(f"## Turn {t.turn}")
            if t.candidates and len(t.candidates) > 1:
                lines.append(f"**Candidates ({len(t.candidates)}):**")
                for i, c in enumerate(t.candidates):
                    marker = "→" if i == t.selected_idx else " "
                    lines.append(f"{marker} {i+1}. {c}")
                lines.append("")
            lines.append(f"**Q:** {t.question}")
            # Show topic classification
            if t.topics_tested:
                behavior = "answer correctly" if t.should_know else "show confusion"
                lines.append(f"**Classification:** Tests `{', '.join(t.topics_tested)}` → {behavior}")
                if t.classification_reasoning:
                    lines.append(f"*{t.classification_reasoning}*")
            if t.initial_answer and t.reflection:
                lines.append(f"**Initial A:** {t.initial_answer}")
                lines.append(f"**Reflection:** {t.reflection}")
                lines.append(f"**Final A:** {t.answer}")
            else:
                lines.append(f"**A:** {t.answer}")
            lines.append("")
            lines.append("| Topic | Guess | Confidence | Reasoning |")
            lines.append("|-------|-------|------------|-----------|")
            for topic, info in t.guess.items():
                known = "✓" if info["known"] else "✗"
                lines.append(f"| {topic} | {known} | {info['confidence']:.0%} | {info['reasoning']} |")
            lines.append(f"\n**Accuracy:** {t.accuracy:.0%}\n")

        return "\n".join(lines)


class RunLogger:
    def __init__(self, base_dir: str = "results"):
        self.base_dir = base_dir
        self.run_dir: Optional[str] = None

    def start_run(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        return self.run_dir

    def save(self, log: RunLog) -> str:
        if not self.run_dir:
            self.start_run()

        filename = f"{log.model.replace('/', '_')}_{log.profile}_n{log.n}.md"
        filepath = os.path.join(self.run_dir, filename)

        with open(filepath, "w") as f:
            f.write(log.to_markdown())

        # Also save JSON for programmatic access
        json_path = filepath.replace(".md", ".json")
        with open(json_path, "w") as f:
            json.dump(asdict(log), f, indent=2)

        return filepath
