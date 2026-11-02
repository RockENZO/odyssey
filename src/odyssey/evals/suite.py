"""A frozen suite of small Python coding tasks.

Each task is a tiny project: starting files, a pytest file that fails at the start, and a
reference solution. The runner gives the agent only the prompt and the starting files. The check
is the project's own tests, with the test files pinned so the agent cannot pass by editing them.

The suite is frozen: ``suite_hash()`` covers every prompt and file, and a test compares it with
``SUITE_HASH``. If you change a task, you are changing what earlier results measured, so bump
``SUITE_VERSION`` and ``SUITE_HASH`` together and do not compare across versions.

Limits: ten short, self-contained Python tasks measure whether a model can use the tools to make
small tests pass. They say little about large codebases, other languages, or ambiguous requests.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field

SUITE_VERSION = "1"
SUITE_HASH = "a66147683e7f082aff8f293b3bd43c40b8622ae70de4114c6d6b90838861dbb8"

# The command that decides whether a task is done. Run in the task's directory.
CHECK = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    files: dict[str, str]                 # starting files, including the pinned tests
    tests: tuple[str, ...]                # which of ``files`` are tests (pinned)
    solution: dict[str, str] = field(default_factory=dict)   # files the reference solution writes


def _task(id, prompt, files, tests, solution):
    return Task(id, prompt, files, tuple(tests), solution)


TASKS: tuple[Task, ...] = (
    _task(
        "off_by_one",
        "`total(n)` in calc.py should return the sum of the integers from 1 to n inclusive, but the tests in tests/test_calc.py fail. Fix calc.py.",
        {
            "calc.py": "def total(n):\n    return sum(range(1, n))\n",
            "tests/test_calc.py": "from calc import total\n\n\ndef test_total():\n    assert total(0) == 0\n    assert total(1) == 1\n    assert total(4) == 10\n",
        },
        ["tests/test_calc.py"],
        {"calc.py": "def total(n):\n    return sum(range(1, n + 1))\n"},
    ),
    _task(
        "fizzbuzz",
        "Implement `fizzbuzz(n)` in fizzbuzz.py as described in its docstring so that tests/test_fizzbuzz.py passes.",
        {
            "fizzbuzz.py": 'def fizzbuzz(n: int) -> str:\n    """Return "FizzBuzz" if n is divisible by 3 and 5, "Fizz" if by 3, "Buzz" if by 5, else str(n)."""\n    raise NotImplementedError\n',
            "tests/test_fizzbuzz.py": 'from fizzbuzz import fizzbuzz\n\n\ndef test_fizzbuzz():\n    assert fizzbuzz(15) == "FizzBuzz"\n    assert fizzbuzz(9) == "Fizz"\n    assert fizzbuzz(10) == "Buzz"\n    assert fizzbuzz(7) == "7"\n',
        },
        ["tests/test_fizzbuzz.py"],
        {"fizzbuzz.py": 'def fizzbuzz(n: int) -> str:\n    """Return "FizzBuzz" if n is divisible by 3 and 5, "Fizz" if by 3, "Buzz" if by 5, else str(n)."""\n    if n % 15 == 0:\n        return "FizzBuzz"\n    if n % 3 == 0:\n        return "Fizz"\n    if n % 5 == 0:\n        return "Buzz"\n    return str(n)\n'},
    ),
    _task(
        "slugify",
        "`slugify` in text.py should turn a title into a URL slug: lowercase, punctuation removed, words joined by single hyphens, no leading or trailing hyphens. Make tests/test_text.py pass.",
        {
            "text.py": 'def slugify(s):\n    return s.lower().replace(" ", "-")\n',
            "tests/test_text.py": 'from text import slugify\n\n\ndef test_slugify():\n    assert slugify("Hello, World!") == "hello-world"\n    assert slugify("  a   b ") == "a-b"\n    assert slugify("Already-Slug") == "already-slug"\n    assert slugify("!!!") == ""\n',
        },
        ["tests/test_text.py"],
        {"text.py": 'import re\n\n\ndef slugify(s):\n    return "-".join(re.findall(r"[a-z0-9]+", s.lower()))\n'},
    ),
    _task(
        "mutable_default",
        "`add` in cart.py shares state between calls, so tests/test_cart.py fails. Fix it without changing its signature for callers that pass `items`.",
        {
            "cart.py": "def add(item, items=[]):\n    items.append(item)\n    return items\n",
            "tests/test_cart.py": 'from cart import add\n\n\ndef test_calls_do_not_share_state():\n    assert add("a") == ["a"]\n    assert add("b") == ["b"]\n\n\ndef test_given_list_is_extended():\n    mine = ["x"]\n    assert add("y", mine) == ["x", "y"]\n    assert mine == ["x", "y"]\n',
        },
        ["tests/test_cart.py"],
        {"cart.py": "def add(item, items=None):\n    if items is None:\n        items = []\n    items.append(item)\n    return items\n"},
    ),
    _task(
        "rename_function",
        "Rename `area_of_circle` in shapes.py to `circle_area` and update everything that uses it, so tests/test_shapes.py passes.",
        {
            "shapes.py": "import math\n\n\ndef area_of_circle(r):\n    return math.pi * r * r\n",
            "report.py": 'from shapes import area_of_circle\n\n\ndef describe(r):\n    return f"circle of radius {r} has area {area_of_circle(r):.2f}"\n',
            "tests/test_shapes.py": 'from report import describe\nfrom shapes import circle_area\n\n\ndef test_circle_area():\n    assert round(circle_area(1), 4) == 3.1416\n\n\ndef test_describe():\n    assert describe(2) == "circle of radius 2 has area 12.57"\n',
        },
        ["tests/test_shapes.py"],
        {
            "shapes.py": "import math\n\n\ndef circle_area(r):\n    return math.pi * r * r\n",
            "report.py": 'from shapes import circle_area\n\n\ndef describe(r):\n    return f"circle of radius {r} has area {circle_area(r):.2f}"\n',
        },
    ),
    _task(
        "parse_kv",
        "Add `parse_kv(text)` to config.py. It turns lines of `key = value` into a dict; blank lines and lines starting with # are ignored, whitespace around keys and values is stripped, and a value may itself contain `=`. Make tests/test_config.py pass.",
        {
            "config.py": '"""Configuration helpers."""\n',
            "tests/test_config.py": 'from config import parse_kv\n\n\ndef test_parse_kv():\n    text = "# comment\\n\\nname = odyssey\\n url=http://x/?a=b \\n"\n    assert parse_kv(text) == {"name": "odyssey", "url": "http://x/?a=b"}\n\n\ndef test_empty():\n    assert parse_kv("") == {}\n',
        },
        ["tests/test_config.py"],
        {"config.py": '"""Configuration helpers."""\n\n\ndef parse_kv(text):\n    result = {}\n    for line in text.splitlines():\n        line = line.strip()\n        if not line or line.startswith("#"):\n            continue\n        key, _, value = line.partition("=")\n        result[key.strip()] = value.strip()\n    return result\n'},
    ),
    _task(
        "stack_max",
        "The Stack class in stack.py needs a `max()` method that returns the largest value currently on the stack, and `pop()` and `max()` must raise IndexError on an empty stack. Make tests/test_stack.py pass.",
        {
            "stack.py": "class Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, value):\n        self._items.append(value)\n\n    def pop(self):\n        return self._items.pop()\n",
            "tests/test_stack.py": 'import pytest\n\nfrom stack import Stack\n\n\ndef test_max_follows_pops():\n    s = Stack()\n    for v in (3, 1, 5, 2):\n        s.push(v)\n    assert s.max() == 5\n    s.pop()\n    s.pop()\n    assert s.max() == 3\n\n\ndef test_empty_raises():\n    s = Stack()\n    with pytest.raises(IndexError):\n        s.pop()\n    with pytest.raises(IndexError):\n        s.max()\n',
        },
        ["tests/test_stack.py"],
        {"stack.py": "class Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, value):\n        self._items.append(value)\n\n    def pop(self):\n        return self._items.pop()\n\n    def max(self):\n        if not self._items:\n            raise IndexError(\"max of empty stack\")\n        return max(self._items)\n"},
    ),
    _task(
        "csv_total",
        "`total_sales(path)` in sales.py sums the `amount` column of a CSV file, but it crashes on files with blank lines. Fix it so tests/test_sales.py passes.",
        {
            "sales.py": 'def total_sales(path):\n    total = 0.0\n    with open(path) as f:\n        header = f.readline().strip().split(",")\n        col = header.index("amount")\n        for line in f:\n            total += float(line.split(",")[col])\n    return total\n',
            "tests/test_sales.py": 'from sales import total_sales\n\n\ndef test_total_sales_skips_blank_lines(tmp_path):\n    p = tmp_path / "s.csv"\n    p.write_text("item,amount\\na,1.5\\n\\nb,2.5\\n\\n")\n    assert total_sales(str(p)) == 4.0\n\n\ndef test_header_only(tmp_path):\n    p = tmp_path / "e.csv"\n    p.write_text("item,amount\\n")\n    assert total_sales(str(p)) == 0.0\n',
        },
        ["tests/test_sales.py"],
        {"sales.py": 'def total_sales(path):\n    total = 0.0\n    with open(path) as f:\n        header = f.readline().strip().split(",")\n        col = header.index("amount")\n        for line in f:\n            if not line.strip():\n                continue\n            total += float(line.split(",")[col])\n    return total\n'},
    ),
    _task(
        "validate_person",
        "`make_person(name, age)` in person.py should raise ValueError when the name is empty (or only spaces), or when age is not an int or is negative. Valid input returns a dict. Make tests/test_person.py pass.",
        {
            "person.py": 'def make_person(name, age):\n    return {"name": name, "age": age}\n',
            "tests/test_person.py": 'import pytest\n\nfrom person import make_person\n\n\ndef test_valid():\n    assert make_person("Ada", 36) == {"name": "Ada", "age": 36}\n\n\n@pytest.mark.parametrize("name,age", [("", 3), ("   ", 3), ("Ada", -1), ("Ada", "3"), ("Ada", 2.5)])\ndef test_invalid(name, age):\n    with pytest.raises(ValueError):\n        make_person(name, age)\n',
        },
        ["tests/test_person.py"],
        {"person.py": 'def make_person(name, age):\n    if not isinstance(name, str) or not name.strip():\n        raise ValueError("name must not be empty")\n    if isinstance(age, bool) or not isinstance(age, int) or age < 0:\n        raise ValueError("age must be a non-negative int")\n    return {"name": name, "age": age}\n'},
    ),
    _task(
        "top_scores",
        "`top_n(scores, n)` in rank.py should return the names of the n highest scores, highest first, with ties broken alphabetically. It currently returns the wrong order. Make tests/test_rank.py pass.",
        {
            "rank.py": "def top_n(scores, n):\n    return [name for name, _ in sorted(scores.items(), key=lambda kv: kv[1])][:n]\n",
            "tests/test_rank.py": 'from rank import top_n\n\n\ndef test_top_n():\n    scores = {"bo": 5, "al": 9, "cy": 5, "di": 1}\n    assert top_n(scores, 3) == ["al", "bo", "cy"]\n    assert top_n(scores, 10) == ["al", "bo", "cy", "di"]\n    assert top_n({}, 2) == []\n',
        },
        ["tests/test_rank.py"],
        {"rank.py": "def top_n(scores, n):\n    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))\n    return [name for name, _ in ordered][:n]\n"},
    ),
)


def suite_hash() -> str:
    """A hash over every prompt, file and solution, so a silent change to a task is caught."""
    payload = [
        {"id": t.id, "prompt": t.prompt, "files": t.files, "tests": list(t.tests), "solution": t.solution}
        for t in TASKS
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def get_tasks(ids: list[str] | None = None) -> list[Task]:
    if not ids:
        return list(TASKS)
    known = {t.id: t for t in TASKS}
    missing = [i for i in ids if i not in known]
    if missing:
        raise KeyError(f"unknown task(s): {', '.join(missing)}")
    return [known[i] for i in ids]
