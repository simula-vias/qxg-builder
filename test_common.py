from common import get_allen

def test_get_allen_equal_no_slack():
    i1 = (0, 1)
    i2 = (0, 1)
    assert get_allen(i1, i2, slack=0) == "E"

    i1 = (0, 2)
    i2 = (0, 1)
    assert get_allen(i1, i2, slack=0) != "E"


def test_get_allen_equal_slack():
    i1 = (0, 1.0)
    i2 = (0.5, 1.5)
    assert get_allen(i1, i2, slack=0.5) == "E"

    i1 = (0, 1.0)
    i2 = (0.0, 2.0)
    assert get_allen(i1, i2, slack=0.5) != "E"


def test_get_allen_meets_slack():
    assert get_allen(
        (0, 1),
        (1.5, 2),
        slack=0.5
    ) == "M"

    assert get_allen(
        (1.5, 2),
        (0, 1),
        slack=0.5
    ) == "MI"


def test_get_allen_before_slack():
    s1, e1 = 0, 1
    slack = 0.5

    assert get_allen(
        (s1, e1),
        (s1-10, s1-slack-0.1),
        slack=slack
    ) == "BI"

    assert get_allen(
        (s1-10, s1-slack-0.1),
        (s1, e1),
        slack=slack
    ) == "B"


def test_get_allen_starts_slack():
    s1, e1 = 0, 1
    slack = 0.5

    assert get_allen(
        (s1, e1),
        (s1+slack, 2*e1),
        slack=slack
    ) == "S"

    assert get_allen(
        (s1, e1),
        (s1-slack, 2*e1),
        slack=slack
    ) == "S"

    assert get_allen(
        (s1+slack, 2*e1),
        (s1, e1),
        slack=slack
    ) == "SI"

    assert get_allen(
        (s1-slack, 2*e1),
        (s1, e1),
        slack=slack
    ) == "SI"



def test_get_allen_overlap_slack():
    s1, e1 = 0, 2
    slack = 0.5

    assert get_allen(
        (s1, e1),
        (s1+slack+0.1, 2*e1),
        slack=slack
    ) == "O"

    assert get_allen(
        (s1+slack+0.1, 2*e1),
        (s1, e1),
        slack=slack
    ) == "OI"


def test_get_allen_during_slack():
    assert get_allen(
        (1, 1.4),
        (0.4, 2),
        slack=0.5
    ) == "D"

    assert get_allen(
        (0.4, 2),
        (1, 1.4),
        slack=0.5
    ) == "DI"


def test_get_allen_finishes_slack():
    s1, e1 = 1, 2
    s2 = 0
    slack = 0.5

    assert get_allen(
        (s1, e1),
        (s2, e1+slack),
        slack=slack
    ) == "F"
    assert get_allen(
        (s1, e1),
        (s2, e1-slack),
        slack=slack
    ) == "F"

    assert get_allen(
        (s2, e1+slack),
        (s1, e1),
        slack=slack
    ) == "FI"
    assert get_allen(
        (s2, e1-slack),
        (s1, e1),
        slack=slack
    ) == "FI"
