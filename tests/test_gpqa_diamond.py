from quantlab.benchmarks.gpqa_diamond import GPQADiamondAdapter, extract_mcq_letter
from quantlab.benchmarks.registry import BenchmarkRegistry


def test_gpqa_diamond_registered():
    assert "gpqa_diamond" in BenchmarkRegistry.available()


def test_extract_mcq_letter():
    assert extract_mcq_letter("Reasoning...\n\nD") == "D"
    assert extract_mcq_letter("Answer: \\boxed{B}") == "B"
    assert extract_mcq_letter("no valid choice here") is None


def test_gpqa_diamond_is_correct():
    adapter = GPQADiamondAdapter()
    assert adapter.is_correct("d", "D")
    assert adapter.is_correct("A", "B") is False
