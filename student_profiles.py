"""
Sample Student Profiles for Diagnostic Search Experiment

This module defines predefined student skill profiles that represent
different archetypes of students across multiple domains. Skills are
domain-agnostic - the system dynamically generates knowledge for any skill.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

# Binary levels for clearer signal (2^n possible states)
LEVELS = ["low", "high"]

# =============================================================================
# Skill Domains - Define different subject areas
# =============================================================================

SKILL_DOMAINS = {
    "computer_science": ["recursion", "pointers", "loops"],
    "math": ["algebra", "calculus", "geometry"],
    "physics": ["mechanics", "thermodynamics", "electromagnetism"],
    "chemistry": ["atomic_structure", "chemical_bonding", "stoichiometry"],
    "biology": ["cell_biology", "genetics", "ecology"],
    "writing": ["grammar", "argumentation", "organization"],
}

# Default skills (can be overridden)
SKILLS = SKILL_DOMAINS["computer_science"]


def set_skill_domain(domain: str) -> List[str]:
    """Set the active skill domain and return its skills."""
    global SKILLS
    if domain not in SKILL_DOMAINS:
        available = ", ".join(SKILL_DOMAINS.keys())
        raise ValueError(f"Unknown domain '{domain}'. Available: {available}")
    SKILLS = SKILL_DOMAINS[domain]
    return SKILLS


def set_custom_skills(skills: List[str]) -> List[str]:
    """Set custom skills (for any domain)."""
    global SKILLS
    if len(skills) < 1:
        raise ValueError("Must provide at least 1 skill")
    SKILLS = skills
    return SKILLS


def get_current_skills() -> List[str]:
    """Get the currently active skills."""
    return SKILLS


@dataclass
class SkillProfile:
    """A student's skill levels across all skills."""
    levels: dict[str, str]  # skill -> "low"/"medium"/"high"
    name: Optional[str] = None  # Optional name for sample profiles
    description: Optional[str] = None  # Optional description
    domain: Optional[str] = None  # Optional domain label

    def __str__(self) -> str:
        prefix = f"[{self.name}] " if self.name else ""
        return prefix + ", ".join(f"{s}: {l}" for s, l in self.levels.items())


# =============================================================================
# Sample Student Profiles by Domain
# =============================================================================

SAMPLE_PROFILES = {
    # Computer Science profiles (8 possible combinations for 3 binary skills)
    "cs_all_low": SkillProfile(
        name="CS All Low",
        description="Struggling across all CS areas",
        domain="computer_science",
        levels={
            "recursion": "low",
            "pointers": "low",
            "loops": "low",
        }
    ),

    "cs_loops_only": SkillProfile(
        name="CS Loops Only",
        description="Only understands loops",
        domain="computer_science",
        levels={
            "recursion": "low",
            "pointers": "low",
            "loops": "high",
        }
    ),

    "cs_pointers_only": SkillProfile(
        name="CS Pointers Only",
        description="Only understands pointers",
        domain="computer_science",
        levels={
            "recursion": "low",
            "pointers": "high",
            "loops": "low",
        }
    ),

    "cs_recursion_only": SkillProfile(
        name="CS Recursion Only",
        description="Only understands recursion",
        domain="computer_science",
        levels={
            "recursion": "high",
            "pointers": "low",
            "loops": "low",
        }
    ),

    "cs_theory_strong": SkillProfile(
        name="CS Theory Strong",
        description="Understands recursion and loops but struggles with pointers",
        domain="computer_science",
        levels={
            "recursion": "high",
            "pointers": "low",
            "loops": "high",
        }
    ),

    "cs_systems_strong": SkillProfile(
        name="CS Systems Strong",
        description="Understands pointers and loops but struggles with recursion",
        domain="computer_science",
        levels={
            "recursion": "low",
            "pointers": "high",
            "loops": "high",
        }
    ),

    "cs_abstract_strong": SkillProfile(
        name="CS Abstract Strong",
        description="Understands recursion and pointers but struggles with loops",
        domain="computer_science",
        levels={
            "recursion": "high",
            "pointers": "high",
            "loops": "low",
        }
    ),

    "cs_all_high": SkillProfile(
        name="CS All High",
        description="Strong across all CS areas",
        domain="computer_science",
        levels={
            "recursion": "high",
            "pointers": "high",
            "loops": "high",
        }
    ),

    # Math profiles
    "math_beginner": SkillProfile(
        name="Math Beginner",
        description="Struggles with advanced topics",
        domain="math",
        levels={
            "algebra": "low",
            "calculus": "low",
            "geometry": "low",
        }
    ),

    "math_calculus_focused": SkillProfile(
        name="Math Calculus Focused",
        description="Strong in algebra and calculus",
        domain="math",
        levels={
            "algebra": "high",
            "calculus": "high",
            "geometry": "low",
        }
    ),

    "math_visual_learner": SkillProfile(
        name="Math Visual Learner",
        description="Great at geometry, struggles with algebra",
        domain="math",
        levels={
            "algebra": "low",
            "calculus": "low",
            "geometry": "high",
        }
    ),

    # Physics profiles
    "physics_beginner": SkillProfile(
        name="Physics Beginner",
        description="Understands basic mechanics only",
        domain="physics",
        levels={
            "mechanics": "high",
            "thermodynamics": "low",
            "electromagnetism": "low",
        }
    ),

    "physics_advanced": SkillProfile(
        name="Physics Advanced",
        description="Strong across all physics areas",
        domain="physics",
        levels={
            "mechanics": "high",
            "thermodynamics": "high",
            "electromagnetism": "high",
        }
    ),
}


def get_sample_profile(name: str) -> SkillProfile:
    """Get a sample student profile by name."""
    if name not in SAMPLE_PROFILES:
        available = ", ".join(SAMPLE_PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return SAMPLE_PROFILES[name]


def list_sample_profiles() -> None:
    """Print all available sample profiles."""
    print("\nAvailable Sample Student Profiles:")
    print("=" * 60)
    for name, profile in SAMPLE_PROFILES.items():
        print(f"\n  {name}:")
        if profile.description:
            print(f"    Description: {profile.description}")
        print("    Skills:")
        for skill, level in profile.levels.items():
            print(f"      {skill}: {level}")


def get_profile_names() -> list[str]:
    """Get list of all profile names."""
    return list(SAMPLE_PROFILES.keys())
