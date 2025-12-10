"""
Comprehensive Student Profiles for Diagnostic Evaluation

30 profiles organized by dimension count (1-10 dimensions, 3 profiles each).
All profiles are realistic - prerequisite knowledge is respected.
"""
from .student import StudentProfile

PROFILES = {
    # =========================================================================
    # 1 DIMENSION (3 profiles)
    # =========================================================================

    "reading_basic": StudentProfile(
        subject="reading",
        knowledge_state={"phonics": True},
    ),
    "reading_none": StudentProfile(
        subject="reading",
        knowledge_state={"phonics": False},
    ),
    "typing_basic": StudentProfile(
        subject="typing",
        knowledge_state={"home_row": True},
    ),

    # =========================================================================
    # 2 DIMENSIONS (3 profiles)
    # =========================================================================

    "arithmetic_complete": StudentProfile(
        subject="arithmetic",
        knowledge_state={"addition": True, "subtraction": True},
    ),
    "arithmetic_partial": StudentProfile(
        subject="arithmetic",
        knowledge_state={"addition": True, "subtraction": False},
    ),
    "logic_intro": StudentProfile(
        subject="logic",
        knowledge_state={"and_or": True, "not": False},
    ),

    # =========================================================================
    # 3 DIMENSIONS (3 profiles)
    # =========================================================================

    "algebra_beginner": StudentProfile(
        subject="algebra",
        knowledge_state={"variables": True, "linear_equations": False, "quadratics": False},
    ),
    "algebra_intermediate": StudentProfile(
        subject="algebra",
        knowledge_state={"variables": True, "linear_equations": True, "quadratics": False},
    ),
    "algebra_advanced": StudentProfile(
        subject="algebra",
        knowledge_state={"variables": True, "linear_equations": True, "quadratics": True},
    ),

    # =========================================================================
    # 4 DIMENSIONS (3 profiles)
    # =========================================================================

    "programming_novice": StudentProfile(
        subject="programming",
        knowledge_state={
            "variables": True,
            "conditionals": True,
            "loops": False,
            "functions": False,
        },
    ),
    "programming_intermediate": StudentProfile(
        subject="programming",
        knowledge_state={
            "variables": True,
            "conditionals": True,
            "loops": True,
            "functions": False,
        },
    ),
    "programming_competent": StudentProfile(
        subject="programming",
        knowledge_state={
            "variables": True,
            "conditionals": True,
            "loops": True,
            "functions": True,
        },
    ),

    # =========================================================================
    # 5 DIMENSIONS (3 profiles)
    # =========================================================================

    "calculus_foundations": StudentProfile(
        subject="calculus",
        knowledge_state={
            "limits": True,
            "continuity": True,
            "derivatives": False,
            "integrals": False,
            "series": False,
        },
    ),
    "calculus_differential": StudentProfile(
        subject="calculus",
        knowledge_state={
            "limits": True,
            "continuity": True,
            "derivatives": True,
            "integrals": False,
            "series": False,
        },
    ),
    "calculus_integral": StudentProfile(
        subject="calculus",
        knowledge_state={
            "limits": True,
            "continuity": True,
            "derivatives": True,
            "integrals": True,
            "series": False,
        },
    ),

    # =========================================================================
    # 6 DIMENSIONS (3 profiles)
    # =========================================================================

    "web_dev_html_only": StudentProfile(
        subject="web_development",
        knowledge_state={
            "html": True,
            "css": False,
            "javascript": False,
            "dom": False,
            "http": False,
            "apis": False,
        },
    ),
    "web_dev_frontend": StudentProfile(
        subject="web_development",
        knowledge_state={
            "html": True,
            "css": True,
            "javascript": True,
            "dom": True,
            "http": False,
            "apis": False,
        },
    ),
    "web_dev_fullstack": StudentProfile(
        subject="web_development",
        knowledge_state={
            "html": True,
            "css": True,
            "javascript": True,
            "dom": True,
            "http": True,
            "apis": True,
        },
    ),

    # =========================================================================
    # 7 DIMENSIONS (3 profiles)
    # =========================================================================

    "statistics_descriptive": StudentProfile(
        subject="statistics",
        knowledge_state={
            "mean_median_mode": True,
            "variance": True,
            "distributions": False,
            "probability": False,
            "hypothesis_testing": False,
            "regression": False,
            "bayesian": False,
        },
    ),
    "statistics_inferential": StudentProfile(
        subject="statistics",
        knowledge_state={
            "mean_median_mode": True,
            "variance": True,
            "distributions": True,
            "probability": True,
            "hypothesis_testing": True,
            "regression": False,
            "bayesian": False,
        },
    ),
    "statistics_advanced": StudentProfile(
        subject="statistics",
        knowledge_state={
            "mean_median_mode": True,
            "variance": True,
            "distributions": True,
            "probability": True,
            "hypothesis_testing": True,
            "regression": True,
            "bayesian": False,
        },
    ),

    # =========================================================================
    # 8 DIMENSIONS (3 profiles)
    # =========================================================================

    "ml_beginner": StudentProfile(
        subject="machine_learning",
        knowledge_state={
            "linear_algebra": True,
            "probability": True,
            "linear_regression": True,
            "logistic_regression": False,
            "decision_trees": False,
            "neural_networks": False,
            "backpropagation": False,
            "regularization": False,
        },
    ),
    "ml_classical": StudentProfile(
        subject="machine_learning",
        knowledge_state={
            "linear_algebra": True,
            "probability": True,
            "linear_regression": True,
            "logistic_regression": True,
            "decision_trees": True,
            "neural_networks": False,
            "backpropagation": False,
            "regularization": True,
        },
    ),
    "ml_deep": StudentProfile(
        subject="machine_learning",
        knowledge_state={
            "linear_algebra": True,
            "probability": True,
            "linear_regression": True,
            "logistic_regression": True,
            "decision_trees": True,
            "neural_networks": True,
            "backpropagation": True,
            "regularization": True,
        },
    ),

    # =========================================================================
    # 9 DIMENSIONS (3 profiles)
    # =========================================================================

    "chemistry_intro": StudentProfile(
        subject="chemistry",
        knowledge_state={
            "atomic_structure": True,
            "periodic_table": True,
            "chemical_bonds": True,
            "balancing_equations": False,
            "stoichiometry": False,
            "acids_bases": False,
            "thermochemistry": False,
            "kinetics": False,
            "equilibrium": False,
        },
    ),
    "chemistry_general": StudentProfile(
        subject="chemistry",
        knowledge_state={
            "atomic_structure": True,
            "periodic_table": True,
            "chemical_bonds": True,
            "balancing_equations": True,
            "stoichiometry": True,
            "acids_bases": True,
            "thermochemistry": False,
            "kinetics": False,
            "equilibrium": False,
        },
    ),
    "chemistry_advanced": StudentProfile(
        subject="chemistry",
        knowledge_state={
            "atomic_structure": True,
            "periodic_table": True,
            "chemical_bonds": True,
            "balancing_equations": True,
            "stoichiometry": True,
            "acids_bases": True,
            "thermochemistry": True,
            "kinetics": True,
            "equilibrium": False,
        },
    ),

    # =========================================================================
    # 10 DIMENSIONS (3 profiles)
    # =========================================================================

    "physics_mechanics_only": StudentProfile(
        subject="physics",
        knowledge_state={
            "kinematics": True,
            "newtons_laws": True,
            "work_energy": True,
            "momentum": False,
            "rotational_motion": False,
            "gravitation": False,
            "oscillations": False,
            "waves": False,
            "thermodynamics": False,
            "electromagnetism": False,
        },
    ),
    "physics_classical": StudentProfile(
        subject="physics",
        knowledge_state={
            "kinematics": True,
            "newtons_laws": True,
            "work_energy": True,
            "momentum": True,
            "rotational_motion": True,
            "gravitation": True,
            "oscillations": True,
            "waves": False,
            "thermodynamics": False,
            "electromagnetism": False,
        },
    ),
    "physics_comprehensive": StudentProfile(
        subject="physics",
        knowledge_state={
            "kinematics": True,
            "newtons_laws": True,
            "work_energy": True,
            "momentum": True,
            "rotational_motion": True,
            "gravitation": True,
            "oscillations": True,
            "waves": True,
            "thermodynamics": True,
            "electromagnetism": False,
        },
    ),
}


def get_profiles_by_dimensions(dims: int) -> dict[str, StudentProfile]:
    """Get all profiles with exactly `dims` dimensions."""
    return {
        name: profile
        for name, profile in PROFILES.items()
        if len(profile.knowledge_state) == dims
    }
