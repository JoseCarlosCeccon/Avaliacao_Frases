from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DELIVERY_DIR = ROOT / "Jose Carlos Ceccon - Henrique Louzada Sara - Trabalho3"
MODELS = (
    ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    ("all-mpnet-base-v2", "all-mpnet-base-v2"),
    (
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ),
    ("BAAI/bge-m3", "BGE-M3"),
)


@pytest.fixture(scope="module")
def corpus() -> set[str]:
    return {
        line.strip()
        for line in (DELIVERY_DIR / "trabalho3-input.txt")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    }


@pytest.mark.parametrize(("model", "file_label"), MODELS)
def test_real_output_has_fifty_verbatim_matches(model, file_label, corpus):
    output_path = DELIVERY_DIR / f"trabalho3-output-{file_label}.txt"
    text = output_path.read_text(encoding="utf-8")

    assert f"Modelo: {model}" in text
    assert text.count("PERGUNTA ") == 10
    lines = [line for line in text.splitlines() if "Distância:" in line]
    assert len(lines) == 50
    for line in lines:
        document = line.split(" | ", 1)[1]
        assert document in corpus


def test_metrics_describe_all_models_questions_and_performance():
    metrics = json.loads(
        (DELIVERY_DIR / "trabalho3-metricas.json").read_text(encoding="utf-8")
    )

    assert metrics["corpus"]["total"] == 1000
    assert metrics["corpus"]["unique"] == 1000
    model_names = {model for model, _file_label in MODELS}
    assert set(metrics["models"]) == model_names
    assert set(metrics["overlap"]) == {str(number) for number in range(1, 11)}
    for model in model_names:
        assert set(metrics["models"][model]["per_question"]) == {
            str(number) for number in range(1, 11)
        }
        performance = metrics["models"][model]["performance"]
        assert performance["indexing_seconds"] >= 0
        assert performance["query_seconds"] >= 0
        assert performance["total_seconds"] == pytest.approx(
            performance["indexing_seconds"] + performance["query_seconds"]
        )


def test_delivery_input_is_identical_to_original():
    assert (DELIVERY_DIR / "trabalho3-input.txt").read_bytes() == (
        ROOT / "trabalho3-input.txt"
    ).read_bytes()
