"""Timetable solver: cost functions and simulated annealing (Phase 3)."""

from .annealing import (
    DEFAULT_MOVES,
    SimulatedAnnealer,
    SolverResult,
    propose_change_room,
    propose_move_course,
    propose_swap,
    solve,
    solve_best_of,
)
from .cost import DEFAULT_SOFT_WEIGHT, CostBreakdown, CostModel, Move, weight_of
from .problem import ProblemIndex
from .timetable import Assignment, Timetable
from .violations import (
    STRUCTURAL_FUNCTIONS,
    VIOLATION_FUNCTIONS,
    constraint_scope,
    violations,
)

__all__ = [
    "DEFAULT_MOVES",
    "DEFAULT_SOFT_WEIGHT",
    "STRUCTURAL_FUNCTIONS",
    "VIOLATION_FUNCTIONS",
    "Assignment",
    "CostBreakdown",
    "CostModel",
    "Move",
    "ProblemIndex",
    "SimulatedAnnealer",
    "SolverResult",
    "Timetable",
    "constraint_scope",
    "propose_change_room",
    "propose_move_course",
    "propose_swap",
    "solve",
    "solve_best_of",
    "violations",
    "weight_of",
]
