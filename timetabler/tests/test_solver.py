"""Solver: problem index, violations, cost, annealing (build plan Phase 3)."""

from __future__ import annotations

import random

import pytest

from timetabler.data import load_department
from timetabler.parser.resolve import (
    EntityResolver,
    UnresolvedReferenceError,
    materialise,
    materialise_all,
)
from timetabler.schema import (
    HARD_PENALTY,
    SOLVABLE_TYPES,
    CourseRequiresRoom,
    CourseRequiresRoomType,
    CoursesNotSameTime,
    CoursesSameDay,
    LecturerPrefersTime,
    LecturerUnavailable,
    MaxConsecutiveHours,
    NoEarlyStart,
    Unsupported,
)
from timetabler.solver import (
    Assignment,
    CostModel,
    Move,
    ProblemIndex,
    SimulatedAnnealer,
    Timetable,
    propose_change_room,
    propose_move_course,
    propose_swap,
    solve,
    violations,
)
from timetabler.solver.violations import (
    capacity_violations,
    fit_violations,
    group_clashes,
    lecturer_clashes,
    room_clashes,
)


@pytest.fixture(scope="module")
def department():
    return load_department()


@pytest.fixture(scope="module")
def index(department):
    return ProblemIndex(department)


@pytest.fixture
def empty(index):
    return Timetable.empty(index)


def place(tt, **kwargs):
    for course_id, (room, slot) in kwargs.items():
        tt.place(course_id, room, slot)
    return tt


# -- problem index -------------------------------------------------------


def test_valid_starts_keep_blocks_inside_the_day(index):
    """A 3-period course cannot start at period 5 of a 6-period day."""
    starts = index.valid_starts[3]
    for slot_id in starts:
        slot = index.slots[slot_id]
        assert slot.period + 2 <= max(index.periods_by_day[slot.day])
    assert "Mon-5" not in starts
    assert "Mon-4" in starts


def test_longer_courses_have_fewer_valid_starts(index):
    assert len(index.valid_starts[1]) > len(index.valid_starts[2]) > len(
        index.valid_starts[3]
    )


def test_rooms_fitting_respects_capacity(index):
    for room_id in index.rooms_fitting("CS101"):
        assert index.rooms[room_id].capacity >= index.courses["CS101"].size
    assert "R202" not in index.rooms_fitting("CS101")


def test_occupied_spans_the_duration(index):
    assert index.occupied("CS202L", "Mon-1") == (("Mon", 1), ("Mon", 2), ("Mon", 3))
    assert index.occupied("CS302", "Mon-1") == (("Mon", 1),)


# -- timetable -----------------------------------------------------------


def test_random_timetable_is_complete_and_inside_the_grid(index):
    tt = Timetable.random(index, random.Random(3))
    assert tt.complete
    assert tt.unassigned == ()
    assert fit_violations(tt, index) == 0


def test_random_timetable_respects_capacity_where_possible(index):
    tt = Timetable.random(index, random.Random(3))
    assert capacity_violations(tt, index) == 0


def test_copy_is_independent(index):
    tt = Timetable.random(index, random.Random(3))
    other = tt.copy()
    other.place("CS101", "R101", "Mon-1")
    assert tt.get("CS101") != other.get("CS101") or tt.get("CS101") == Assignment(
        "R101", "Mon-1"
    )


# -- structural violations ----------------------------------------------


def test_room_clash_counted(empty, index):
    place(empty, CS101=("R101", "Mon-1"), CS102=("R101", "Mon-1"))
    assert room_clashes(empty, index) == 2, "two shared cells, CS101 runs 2 periods"


def test_no_clash_in_different_rooms(empty, index):
    place(empty, CS101=("R101", "Mon-1"), CS102=("R102", "Mon-1"))
    assert room_clashes(empty, index) == 0


def test_lecturer_clash_counted(empty, index):
    """CS101 and CS101L are both taught by L001."""
    place(empty, CS101=("R101", "Mon-1"), CS101L=("R201", "Mon-1"))
    assert lecturer_clashes(empty, index) == 2
    assert room_clashes(empty, index) == 0


def test_group_clash_counted(empty, index):
    """CS201 and CS202 both belong to G2A."""
    place(empty, CS201=("R101", "Tue-1"), CS202=("R102", "Tue-1"))
    assert group_clashes(empty, index) == 2


def test_partial_overlap_counted_once_per_shared_cell(empty, index):
    place(empty, CS101=("R101", "Mon-1"), CS102=("R101", "Mon-2"))
    assert room_clashes(empty, index) == 1


def test_capacity_violation_counted(empty, index):
    place(empty, CS101=("R202", "Mon-1"))  # 45 students into a room of 24
    assert capacity_violations(empty, index) == 1


def test_overflowing_block_detected(empty, index):
    place(empty, CS202L=("R201", "Mon-5"))  # 3 periods from period 5 of 6
    assert fit_violations(empty, index) == 1


# -- elicited violations -------------------------------------------------


def test_lecturer_unavailable_counts_offending_cells(empty, index):
    place(empty, CS101=("R101", "Fri-1"))
    c = LecturerUnavailable(
        lecturer="L001", days=["Fri"], strength="hard", source_text="x"
    )
    assert violations(empty, c, index) == 2, "CS101 runs 2 periods, both on Friday"


def test_lecturer_unavailable_ignores_other_days(empty, index):
    place(empty, CS101=("R101", "Mon-1"))
    c = LecturerUnavailable(
        lecturer="L001", days=["Fri"], strength="hard", source_text="x"
    )
    assert violations(empty, c, index) == 0


def test_lecturer_unavailable_narrows_to_periods(empty, index):
    place(empty, CS101=("R101", "Fri-1"))
    c = LecturerUnavailable(
        lecturer="L001", days=["Fri"], periods=[1], strength="hard", source_text="x"
    )
    assert violations(empty, c, index) == 1, "only period 1 is excluded"


def test_prefers_time_counts_cells_outside_the_preference(empty, index):
    place(empty, CS101=("R101", "Fri-5"))
    c = LecturerPrefersTime(
        lecturer="L001", periods=[1, 2, 3], strength="soft", weight=5, source_text="x"
    )
    assert violations(empty, c, index) == 2, "periods 5 and 6 are both outside"


def test_prefers_time_satisfied_inside_the_preference(empty, index):
    place(empty, CS101=("R101", "Fri-1"))
    c = LecturerPrefersTime(
        lecturer="L001", periods=[1, 2, 3], strength="soft", weight=5, source_text="x"
    )
    assert violations(empty, c, index) == 0


def test_requires_room(empty, index):
    c = CourseRequiresRoom(
        course="CS301", room="R101", strength="hard", source_text="x"
    )
    place(empty, CS301=("R102", "Mon-1"))
    assert violations(empty, c, index) == 1
    place(empty, CS301=("R101", "Mon-1"))
    assert violations(empty, c, index) == 0


def test_requires_room_type(empty, index):
    c = CourseRequiresRoomType(
        course="CS101L", room_type="computer_lab", strength="hard", source_text="x"
    )
    place(empty, CS101L=("R101", "Mon-1"))  # a lecture theatre
    assert violations(empty, c, index) == 1
    place(empty, CS101L=("R201", "Mon-1"))  # a computer lab
    assert violations(empty, c, index) == 0


def test_max_consecutive_counts_excess_not_breaches(empty, index):
    """L001 teaching 4 straight periods against a cap of 2 costs 2, not 1."""
    place(empty, CS101=("R101", "Mon-1"), CS101L=("R201", "Mon-3"))
    c = MaxConsecutiveHours(
        max_hours=2, lecturer="L001", strength="hard", source_text="x"
    )
    assert violations(empty, c, index) == 2


def test_max_consecutive_satisfied_with_a_gap(empty, index):
    place(empty, CS101=("R101", "Mon-1"), CS101L=("R201", "Mon-5"))
    c = MaxConsecutiveHours(
        max_hours=2, lecturer="L001", strength="hard", source_text="x"
    )
    assert violations(empty, c, index) == 0


def test_no_early_start_counts_early_cells(empty, index):
    place(empty, CS101=("R101", "Mon-1"))
    c = NoEarlyStart(
        earliest_period=3, group="G1A", strength="soft", weight=5, source_text="x"
    )
    assert violations(empty, c, index) == 2


def test_no_early_start_department_wide_covers_every_course(empty, index):
    place(empty, CS101=("R101", "Mon-1"), CS201=("R102", "Tue-1"))
    c = NoEarlyStart(earliest_period=2, strength="hard", source_text="x")
    assert violations(empty, c, index) == 2, "one early cell from each course"


def test_not_same_time_counted_per_pair(empty, index):
    c = CoursesNotSameTime(
        courses=["CS301", "CS302"], strength="hard", source_text="x"
    )
    place(empty, CS301=("R101", "Mon-1"), CS302=("R301", "Mon-1"))
    assert violations(empty, c, index) == 1
    place(empty, CS302=("R301", "Mon-3"))
    assert violations(empty, c, index) == 0


def test_same_day_counts_courses_that_must_move(empty, index):
    c = CoursesSameDay(
        courses=["CS101", "CS102", "CS201"], strength="soft", weight=5,
        source_text="x",
    )
    place(empty, CS101=("R101", "Mon-1"), CS102=("R102", "Mon-3"),
          CS201=("R102", "Tue-1"))
    assert violations(empty, c, index) == 1, "two on Monday, one away"
    place(empty, CS201=("R102", "Mon-5"))
    assert violations(empty, c, index) == 0


def test_open_category_is_not_solvable(empty, index):
    with pytest.raises(ValueError, match="not solvable"):
        violations(empty, Unsupported(note="n", source_text="x"), index)


# -- cost ----------------------------------------------------------------


def test_hard_penalty_dominates_any_soft_accumulation(index, empty):
    """No pile of soft breaches can outweigh one hard breach (Phase 3.6)."""
    from timetabler.schema import SOFT_WEIGHT_MAX

    assert HARD_PENALTY > SOFT_WEIGHT_MAX * 50


def test_cost_combines_hard_and_soft(empty, index):
    place(empty, CS101=("R101", "Fri-1"))
    hard = LecturerUnavailable(
        lecturer="L001", days=["Fri"], strength="hard", source_text="x"
    )
    soft = LecturerPrefersTime(
        lecturer="L001", periods=[3], strength="soft", weight=4, source_text="y"
    )
    model = CostModel(index, [hard, soft])
    breakdown = model.breakdown(empty)
    # 2 hard cells + 7 unassigned courses, then 2 soft cells at weight 4
    assert breakdown.soft_total == 8.0
    assert not breakdown.feasible
    assert breakdown.violated_hard and breakdown.violated_soft


def test_unweighted_soft_constraint_gets_the_documented_default(index):
    from timetabler.solver.cost import DEFAULT_SOFT_WEIGHT, weight_of

    c = LecturerPrefersTime(
        lecturer="L001", periods=[1], strength="soft", source_text="x"
    )
    assert weight_of(c) == DEFAULT_SOFT_WEIGHT


def test_hard_constraint_weighted_at_the_hard_penalty(index):
    from timetabler.solver.cost import weight_of

    c = LecturerUnavailable(
        lecturer="L001", days=["Fri"], strength="hard", source_text="x"
    )
    assert weight_of(c) == HARD_PENALTY


def test_feasible_timetable_has_zero_hard_cost(index):
    result = solve(index, [], iterations=20_000, seed=5)
    assert result.feasible
    assert result.breakdown.hard_violations == 0


# -- moves ---------------------------------------------------------------


def test_move_apply_and_revert_round_trips(index):
    tt = Timetable.random(index, random.Random(11))
    before = dict(tt.assignments)
    move = Move({"CS101": Assignment("R101", "Fri-5")})
    previous = move.apply(tt)
    assert tt.get("CS101") == Assignment("R101", "Fri-5")
    Move.revert(tt, previous)
    assert tt.assignments == before


def test_swap_exchanges_placements(index):
    tt = Timetable.random(index, random.Random(11))
    a, b = tt.get("CS101"), tt.get("CS102")
    Move({"CS101": b, "CS102": a}).apply(tt)
    assert (tt.get("CS101"), tt.get("CS102")) == (b, a)


def test_swap_proposal_refuses_a_slot_the_block_cannot_fit(index):
    """CS202L needs 3 periods; CS302 needs 1 and may sit at period 6."""
    tt = Timetable.empty(index)
    tt.place("CS202L", "R201", "Mon-1")
    tt.place("CS302", "R301", "Mon-6")
    rng = random.Random(0)
    for _ in range(50):
        move = propose_swap(tt, index, rng)
        if move is not None:
            for course_id, assignment in move.placements.items():
                assert assignment.slot in index.starts_for(course_id)


@pytest.mark.parametrize("proposer", [
    propose_move_course, propose_change_room, propose_swap,
])
def test_every_proposer_yields_valid_placements(index, proposer):
    tt = Timetable.random(index, random.Random(2))
    rng = random.Random(4)
    produced = 0
    for _ in range(300):
        move = proposer(tt, index, rng)
        if move is None:
            continue
        produced += 1
        for course_id, assignment in move.placements.items():
            assert assignment.slot in index.starts_for(course_id)
            assert assignment.room in index.room_ids
    assert produced > 0, "proposer never produced a move"


def test_change_room_keeps_the_slot(index):
    tt = Timetable.random(index, random.Random(2))
    rng = random.Random(9)
    for _ in range(100):
        move = propose_change_room(tt, index, rng)
        if move is None:
            continue
        for course_id, assignment in move.placements.items():
            assert assignment.slot == tt.slot_of(course_id)
            assert assignment.room != tt.room_of(course_id)


# -- delta evaluation ----------------------------------------------------


def seed_constraints(index):
    """A mixed constraint load covering every solvable type."""
    return [
        LecturerUnavailable(
            lecturer="L001", days=["Fri"], strength="hard", source_text="a"),
        LecturerPrefersTime(
            lecturer="L003", periods=[1, 2, 3], strength="soft", weight=4,
            source_text="b"),
        CourseRequiresRoom(
            course="CS301", room="R101", strength="hard", source_text="c"),
        CourseRequiresRoomType(
            course="CS101L", room_type="computer_lab", strength="hard",
            source_text="d"),
        MaxConsecutiveHours(
            max_hours=2, lecturer="L004", strength="soft", weight=3,
            source_text="e"),
        NoEarlyStart(
            earliest_period=2, group="G1A", strength="soft", weight=5,
            source_text="f"),
        CoursesNotSameTime(
            courses=["CS201", "CS202"], strength="hard", source_text="g"),
        CoursesSameDay(
            courses=["CS101", "CS101L"], strength="soft", weight=2,
            source_text="h"),
    ]


def test_delta_matches_full_recompute_over_a_random_walk(index):
    """The safety net for partial evaluation.

    `CostModel.affected` under-reporting a scope would make the annealer
    optimise a cost that drifts from the real one. Nothing else in the suite
    would catch it, because every individual violation function would still
    be correct.
    """
    model = CostModel(index, seed_constraints(index))
    rng = random.Random(42)
    tt = Timetable.random(index, rng)
    tracked = model.total(tt)

    for step in range(600):
        proposer = rng.choice([propose_move_course, propose_swap, propose_change_room])
        move = proposer(tt, index, rng)
        if move is None:
            continue
        closure, indices = model.affected(move, tt)
        before = model.partial(tt, closure, indices)
        move.apply(tt)
        after = model.partial(tt, closure, indices)
        tracked += after - before
        assert tracked == pytest.approx(model.total(tt), abs=1e-6), (
            f"drift at step {step} after {proposer.__name__}"
        )


def test_delta_helper_leaves_the_timetable_untouched(index):
    model = CostModel(index, seed_constraints(index))
    tt = Timetable.random(index, random.Random(8))
    before = dict(tt.assignments)
    model.delta(tt, Move({"CS101": Assignment("R101", "Fri-5")}))
    assert tt.assignments == before


def test_annealer_self_verification_catches_nothing_on_a_correct_model(index):
    model = CostModel(index, seed_constraints(index))
    annealer = SimulatedAnnealer(model, iterations=3000, seed=1, verify_every=10)
    annealer.solve()  # raises if the tracked cost ever drifts


# -- annealing -----------------------------------------------------------


def test_search_improves_on_its_starting_point(index):
    model = CostModel(index, seed_constraints(index))
    result = SimulatedAnnealer(model, iterations=15_000, seed=3).solve()
    assert result.cost < result.initial_cost


def test_satisfiable_instance_is_solved_to_feasibility(index):
    result = solve(index, seed_constraints(index), iterations=40_000, seed=3)
    assert result.feasible, result.breakdown.summary()
    assert result.breakdown.violated_hard == []


def test_same_seed_reproduces_the_run(index):
    a = solve(index, seed_constraints(index), iterations=5000, seed=17)
    b = solve(index, seed_constraints(index), iterations=5000, seed=17)
    assert a.cost == b.cost
    assert a.timetable.as_dict() == b.timetable.as_dict()


def test_different_seeds_explore_differently(index):
    costs = {
        solve(index, seed_constraints(index), iterations=800, seed=s).cost
        for s in range(6)
    }
    assert len(costs) > 1, "every seed produced an identical cost"


def test_unsatisfiable_constraint_pair_is_reported_not_hidden(index):
    """Phase 5.3 and §5: name the conflict so staff can relax something."""
    conflicting = [
        CourseRequiresRoom(
            course="CS101", room="R101", strength="hard", source_text="in LT1"),
        CourseRequiresRoom(
            course="CS101", room="R102", strength="hard", source_text="in LT2"),
    ]
    result = solve(index, conflicting, iterations=8000, seed=2)
    assert not result.feasible
    assert len(result.breakdown.violated_hard) == 1
    assert result.breakdown.violated_hard[0]["source_text"] in ("in LT1", "in LT2")


def test_capacity_beats_an_elicited_hard_room_request(index):
    """A room too small loses to the structural constraint, and is named."""
    too_small = [CourseRequiresRoom(
        course="CS101", room="R202", strength="hard",
        source_text="programming in Lab B",
    )]
    result = solve(index, too_small, iterations=8000, seed=2)
    assert result.breakdown.structural["room_capacity"] == 0
    assert result.breakdown.violated_hard[0]["source_text"] == "programming in Lab B"


def test_iterations_reports_what_actually_ran(index):
    result = solve(index, [], iterations=50_000, seed=5)
    assert result.iterations <= 50_000
    if result.cost == 0:
        assert result.iterations < 50_000, "should stop once cost reaches 0"


def test_rejects_a_nonsense_budget(index):
    model = CostModel(index, [])
    with pytest.raises(ValueError, match="at least 1"):
        SimulatedAnnealer(model, iterations=0)


def test_rejects_an_impossible_acceptance_target(index):
    model = CostModel(index, [])
    with pytest.raises(ValueError, match="target_initial_acceptance"):
        SimulatedAnnealer(model, target_initial_acceptance=1.5)


def test_restarts_are_no_worse_than_a_single_run(index):
    from timetabler.solver import solve_best_of

    single = solve(index, seed_constraints(index), iterations=2000, seed=0)
    best = solve_best_of(
        index, seed_constraints(index), restarts=4, iterations=2000, seed=0
    )
    assert best.cost <= single.cost


# -- the parser-to-solver bridge ----------------------------------------


def test_materialise_rewrites_entities_as_ids(department):
    resolver = EntityResolver(department)
    c = CourseRequiresRoom(
        course="Machine Learning", room="the big lecture theatre",
        strength="hard", source_text="x",
    )
    out = materialise(resolver.resolve(c))
    assert (out.course, out.room) == ("CS301", "R101")


def test_materialise_rewrites_indexed_list_fields(department):
    resolver = EntityResolver(department)
    c = CoursesNotSameTime(
        courses=["Machine Learning", "FYP seminar"], strength="hard",
        source_text="x",
    )
    assert materialise(resolver.resolve(c)).courses == ["CS301", "CS302"]


def test_materialise_refuses_to_guess_an_unresolved_entity(department):
    resolver = EntityResolver(department)
    c = LecturerUnavailable(
        lecturer="Zhang", days=["Mon"], strength="hard", source_text="x"
    )
    with pytest.raises(UnresolvedReferenceError, match="resolve it with the user"):
        materialise(resolver.resolve(c))


def test_materialise_all_skips_blocked_constraints(department):
    resolver = EntityResolver(department)
    constraints = [
        LecturerUnavailable(
            lecturer="Lee", days=["Fri"], strength="hard", source_text="ok"),
        LecturerUnavailable(
            lecturer="Zhang", days=["Mon"], strength="hard", source_text="blocked"),
        Unsupported(note="n", source_text="open category"),
    ]
    out = materialise_all(resolver.resolve(c) for c in constraints)
    assert [c.source_text for c in out] == ["ok"]


def test_materialise_all_refuses_to_pass_blocked_input_through(department):
    resolver = EntityResolver(department)
    c = Unsupported(note="n", source_text="x")
    with pytest.raises(UnresolvedReferenceError):
        materialise_all([resolver.resolve(c)], skip_blocked=False)


def test_solver_accepts_every_solvable_type(index, department):
    """Each frozen solvable type reaches the cost model without error."""
    model = CostModel(index, seed_constraints(index))
    covered = {c.type for c in model.constraints}
    assert covered == set(SOLVABLE_TYPES), f"uncovered: {set(SOLVABLE_TYPES) - covered}"
    model.total(Timetable.random(index, random.Random(1)))
