"""Utilities to process annotation in metadata."""

import enum
import functools
import operator
import itertools


class Action(enum.Flag):
    """Annotated surgical actions."""

    # Common
    IDLE = enum.auto()
    IN_OUT_WOUND = enum.auto()
    RECOVERY_PRODUCTIVE_ADJUSTMENT = enum.auto()
    UNPRODUCTIVE_MANIPULATION = enum.auto()
    # CCC specific
    CUT = enum.auto()
    GRASP = enum.auto()
    TEAR = enum.auto()
    # Phaco specific
    CHOP = enum.auto()
    ENGAGE = enum.auto()
    PULL = enum.auto()
    ROTATION = enum.auto()
    SCULPT = enum.auto()
    SEPARATE = enum.auto()
    HOOK = enum.auto()
    SUCTION = enum.auto()
    HOOK_AND_SUCTION = enum.auto()

    # Composites
    ALL = (
        IDLE
        | IN_OUT_WOUND
        | RECOVERY_PRODUCTIVE_ADJUSTMENT
        | UNPRODUCTIVE_MANIPULATION
        | CUT
        | GRASP
        | TEAR
        | CHOP
        | ENGAGE
        | PULL
        | ROTATION
        | SCULPT
        | SEPARATE
        | HOOK
        | SUCTION
        | HOOK_AND_SUCTION
    )
    ALL_CCC = (
        IDLE
        | IN_OUT_WOUND
        | RECOVERY_PRODUCTIVE_ADJUSTMENT
        | UNPRODUCTIVE_MANIPULATION
        | CUT
        | GRASP
        | TEAR
    )
    ALL_PHACO = (
        IDLE
        | IN_OUT_WOUND
        | RECOVERY_PRODUCTIVE_ADJUSTMENT
        | UNPRODUCTIVE_MANIPULATION
        | CHOP
        | ENGAGE
        | PULL
        | ROTATION
        | SCULPT
        | SEPARATE
        | HOOK
        | SUCTION
        | HOOK_AND_SUCTION
    )


ACTION_FROM_TEXT: dict[str, Action] = {
    "Idle": Action.IDLE,
    "In-Out wound": Action.IN_OUT_WOUND,
    "Recovery / Productive Adjustment": Action.RECOVERY_PRODUCTIVE_ADJUSTMENT,
    "Unproductive Manipulation": Action.UNPRODUCTIVE_MANIPULATION,
    "Cut": Action.CUT,
    "Grasp": Action.GRASP,
    "Tear": Action.TEAR,
    "Chop": Action.CHOP,
    "Engage": Action.ENGAGE,
    "Hook": Action.HOOK,
    "Hook and Suction": Action.HOOK_AND_SUCTION,
    "Pull": Action.PULL,
    "Rotation": Action.ROTATION,
    "Sculpt": Action.SCULPT,
    "Separate": Action.SEPARATE,
    "Suction": Action.SUCTION,
}


TEXT_FROM_ACTION: dict[Action, str] = {
    action: text for text, action in ACTION_FROM_TEXT.items()
}


COMPOSITE_ACTION_FROM_TEXT: dict[str, Action] = {
    "All": Action.ALL,
    "All CCC": Action.ALL_CCC,
    "All Phaco": Action.ALL_PHACO,
}


def groups_from_actions(actions: Action) -> list[Action]:
    """Return action groups where each action belongs to an individual group."""
    return [action for action in actions]


def actions_from_groups(action_groups: list[Action]) -> Action:
    """Return collection of actions involved in the action groups."""
    return functools.reduce(operator.or_, action_groups)


def action_group_mapping(action_groups: list[Action]) -> dict[Action, Action]:
    """Return the mapping from action to its group."""
    return {action: group for group in action_groups for action in group}


def action_group_label_mapping(action_groups: list[Action]) -> dict[Action, int]:
    """Return the mapping from action groups to their labels."""
    return {group: ind for ind, group in enumerate(action_groups)}


def action_group_text_mapping(action_groups: list[Action]) -> dict[Action, str]:
    """Return the mapping from action groups to their texts."""
    return {
        group: ", ".join(TEXT_FROM_ACTION[action] for action in group)
        for group in action_groups
    }


def actions_from_texts(texts: list[str]) -> Action:
    """Return actions of texts."""
    assert len(texts) > 0
    if len(texts) == 1 and texts[0] in COMPOSITE_ACTION_FROM_TEXT:
        return COMPOSITE_ACTION_FROM_TEXT[texts[0]]
    else:
        return functools.reduce(
            operator.or_, [ACTION_FROM_TEXT[text] for text in texts]
        )


def validate_action_groups(action_groups: list[Action]) -> None:
    """Raise exception if action groups are no valid."""
    assert all(len(group) > 0 for group in action_groups)
    assert all(
        len(grp_1 & grp_2) == 0
        for grp_1, grp_2 in itertools.combinations(action_groups, 2)
    )


def action_groups_from_texts(texts: list[list[str]]) -> list[Action]:
    """Return action groups of texts."""
    assert len(texts) > 0
    assert all(len(_texts) > 0 for _texts in texts)
    action_groups = [
        functools.reduce(operator.or_, [ACTION_FROM_TEXT[text] for text in _texts])
        for _texts in texts
    ]
    validate_action_groups(action_groups)
    return action_groups


def action_groups_from_arguments(
    actions: list[str], action_groups: list[list[str]] | None
) -> list[Action]:
    """Return action groups of the parsed command-line arguments."""
    if action_groups is not None:
        return action_groups_from_texts(action_groups)
    else:
        return groups_from_actions(actions_from_texts(actions))


class Phase(enum.Flag):
    """Annotated surgical phases."""

    CCC = enum.auto()
    PHACO = enum.auto()
    # Composites
    ALL = CCC | PHACO


PHASE_FROM_TEXT: dict[str, Phase] = {
    "ccc": Phase.CCC,
    "phaco": Phase.PHACO,
}
TEXT_FROM_PHASE: dict[Phase, str] = {
    phase: text for text, phase in PHASE_FROM_TEXT.items()
}


COMPOSITE_PHASE_FROM_TEXT: dict[str, Phase] = {
    "all": Phase.ALL,
}


def phases_from_texts(texts: list[str]) -> Phase:
    """Return phases of texts."""
    assert len(texts) > 0
    if len(texts) == 1 and texts[0] in COMPOSITE_PHASE_FROM_TEXT:
        return COMPOSITE_PHASE_FROM_TEXT[texts[0]]
    else:
        return functools.reduce(operator.or_, [PHASE_FROM_TEXT[text] for text in texts])


class ActionScore(enum.Flag):
    """Perfromance score of action."""

    ONE = enum.auto()
    TWO = enum.auto()
    THREE = enum.auto()
    FOUR = enum.auto()
    FIVE = enum.auto()
    NAN = enum.auto()
    # Composites
    ALL = ONE | TWO | THREE | FOUR | FIVE | NAN


SCORE_FROM_TEXT: dict[str, ActionScore] = {
    "one": ActionScore.ONE,
    "two": ActionScore.TWO,
    "three": ActionScore.THREE,
    "four": ActionScore.FOUR,
    "five": ActionScore.FIVE,
    "nan": ActionScore.NAN,
}
COMPOSITE_SCORE_FROM_TEXT: dict[str, ActionScore] = {
    "all": ActionScore.ALL,
}


def scores_from_texts(texts: list[str]) -> ActionScore:
    """Return action scores of texts."""
    assert len(texts) > 0
    if len(texts) == 1 and texts[0] in COMPOSITE_SCORE_FROM_TEXT:
        return COMPOSITE_SCORE_FROM_TEXT[texts[0]]
    else:
        return functools.reduce(operator.or_, [SCORE_FROM_TEXT[text] for text in texts])


NUMBER_FROM_SCORE: dict[ActionScore, float] = {
    ActionScore.ONE: 1.0,
    ActionScore.TWO: 2.0,
    ActionScore.THREE: 3.0,
    ActionScore.FOUR: 4.0,
    ActionScore.FIVE: 5.0,
}


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
