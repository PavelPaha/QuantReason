from quantlab.benchmarks.aime26 import AIME26Adapter
from quantlab.benchmarks.registry import BenchmarkRegistry


def test_aime26_registered():
    assert "aime26" in BenchmarkRegistry.available()


def test_aime26_extract_answer_boxed_only():
    adapter = AIME26Adapter()
    assert adapter.extract_answer("Reasoning... \\boxed{277}") == "277"
    assert adapter.extract_answer("First \\boxed{1} then \\boxed{42}") == "42"
    assert adapter.extract_answer("The answer is 277 without a box.") is None
    assert adapter.extract_answer("no answer here") is None


def test_aime26_is_correct_integer_match():
    adapter = AIME26Adapter()
    assert adapter.is_correct("277", "277")
    assert adapter.is_correct("0277", "277")
    assert adapter.is_correct(None, "277") is False
    assert adapter.is_correct("278", "277") is False
