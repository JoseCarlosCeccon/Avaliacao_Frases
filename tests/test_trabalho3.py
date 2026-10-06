from __future__ import annotations

import sys
import json
import os
from pathlib import Path

import pytest


DELIVERY_DIR = (
    Path(__file__).resolve().parents[1]
    / "Jose Carlos Ceccon - Henrique Louzada Sara - Trabalho3"
)
sys.path.insert(0, str(DELIVERY_DIR))

import trabalho3
from trabalho3 import (
    Match,
    ModelRun,
    QuestionResult,
    atomic_write_text,
    build_reference_catalog,
    compare_results,
    format_output,
    main,
    output_filename,
    query_collection,
    read_corpus,
    run_model,
    stable_ids,
)


def test_read_corpus_preserves_utf8_order_and_ignores_blank_lines(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text(
        "saúde digital\n\n  \ncomputação em nuvem\n", encoding="utf-8"
    )

    assert read_corpus(source) == ["saúde digital", "computação em nuvem"]


def test_read_corpus_rejects_empty_input(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="nenhuma frase"):
        read_corpus(source)


def test_stable_ids_are_unique_and_zero_padded():
    assert stable_ids(3) == ["frase-000001", "frase-000002", "frase-000003"]


def sample_results(question_count: int, matches_per_question: int):
    return [
        QuestionResult(
            number=number,
            question=(
                "Como a inteligência artificial está mudando o mercado de trabalho?"
                if number == 1
                else f"Pergunta de teste {number}?"
            ),
            matches=tuple(
                Match(document=f"Frase {number}-{rank}", distance=rank / 10)
                for rank in range(1, matches_per_question + 1)
            ),
        )
        for number in range(1, question_count + 1)
    ]


def test_format_output_contains_ten_questions_and_five_ranked_matches():
    results = sample_results(question_count=10, matches_per_question=5)

    text = format_output("all-MiniLM-L6-v2", 1000, results)

    assert "Modelo: all-MiniLM-L6-v2" in text
    assert "Frases indexadas: 1000" in text
    assert text.count("PERGUNTA ") == 10
    assert text.count("Distância:") == 50
    assert "1. Como a inteligência artificial" in text


class FakeCollection:
    def __init__(self, response=None):
        self.response = response
        self.requested_n_results = None
        self.added_ids = None
        self.added_documents = None
        self.added_batches = []

    def add(self, *, ids, documents):
        self.added_ids = ids
        self.added_documents = (self.added_documents or []) + list(documents)
        self.added_batches.append((list(ids), list(documents)))

    def query(self, *, query_texts, n_results, include):
        self.requested_n_results = n_results
        if self.response is not None:
            return self.response
        documents = [self.added_documents[:n_results] for _ in query_texts]
        distances = [
            [index / 10 for index in range(n_results)] for _ in query_texts
        ]
        return {"documents": documents, "distances": distances}


def test_query_collection_maps_documents_and_distances_in_rank_order():
    collection = FakeCollection(
        {
            "documents": [["frase A", "frase B"]],
            "distances": [[0.10, 0.25]],
        }
    )

    result = query_collection(collection, ["pergunta"], n_results=2)

    assert result[0].matches == (
        Match("frase A", 0.10),
        Match("frase B", 0.25),
    )
    assert collection.requested_n_results == 2


def test_run_model_caps_n_results_to_corpus_size(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(trabalho3, "create_collection", lambda _model: collection)

    run = run_model("all-MiniLM-L6-v2", ["A", "B"], ["pergunta"])

    assert len(run.results[0].matches) == 2
    assert collection.requested_n_results == 2


def test_run_model_adds_large_corpus_in_bounded_batches(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(trabalho3, "create_collection", lambda _model: collection)
    monkeypatch.setattr(trabalho3, "ADD_BATCH_SIZE", 2)

    run_model("all-MiniLM-L6-v2", ["A", "B", "C", "D", "E"], ["pergunta"])

    assert [len(documents) for _ids, documents in collection.added_batches] == [2, 2, 1]


def test_run_model_measures_index_query_and_total_time(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(trabalho3, "create_collection", lambda _model: collection)
    ticks = iter([10.0, 12.5, 13.25])

    run = run_model(
        "all-MiniLM-L6-v2",
        ["A", "B"],
        ["pergunta"],
        clock=lambda: next(ticks),
    )

    assert run.indexing_seconds == pytest.approx(2.5)
    assert run.query_seconds == pytest.approx(0.75)
    assert run.total_seconds == pytest.approx(3.25)


def test_models_include_originals_and_two_free_multilingual_models():
    assert trabalho3.MODELS == (
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-m3",
    )


def test_hugging_face_uses_http_fallback_for_large_model_downloads():
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_output_filename_uses_readable_bge_alias():
    assert output_filename("BAAI/bge-m3") == "trabalho3-output-BGE-M3.txt"
    assert output_filename("all-MiniLM-L6-v2") == (
        "trabalho3-output-all-MiniLM-L6-v2.txt"
    )


def test_format_output_reports_actual_count_for_short_corpus():
    results = sample_results(question_count=1, matches_per_question=2)

    text = format_output("modelo", 2, results)

    assert "Resultados por pergunta: 2" in text


def test_query_collection_rejects_misaligned_chromadb_response():
    collection = FakeCollection(
        {"documents": [["A", "B"]], "distances": [[0.1]]}
    )

    with pytest.raises(ValueError, match="inconsistente"):
        query_collection(collection, ["pergunta"], n_results=2)


def test_atomic_write_replaces_destination_without_temp_file(tmp_path):
    target = tmp_path / "saida.txt"
    target.write_text("antigo", encoding="utf-8")

    atomic_write_text(target, "novo\n")

    assert target.read_text(encoding="utf-8") == "novo\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_compare_results_calculates_recall_mrr_and_overlap():
    model_a = [
        QuestionResult(1, "Q", (Match("R1", 0.1), Match("X", 0.2)))
    ]
    model_b = [
        QuestionResult(1, "Q", (Match("X", 0.1), Match("R1", 0.2)))
    ]

    metrics = compare_results(
        {"A": model_a, "B": model_b}, {1: {"R1", "R2"}}
    )

    assert metrics["models"]["A"]["recall_at_5"] == pytest.approx(0.5)
    assert metrics["models"]["A"]["mrr"] == pytest.approx(1.0)
    assert metrics["models"]["B"]["mrr"] == pytest.approx(0.5)
    assert metrics["overlap"]["1"] == 2


def test_build_reference_catalog_assigns_clear_topic_sentences():
    documents = [
        "Profissões inteiras estão sendo reinventadas por programas.",
        "Criptografar informações sensíveis impede leituras não autorizadas.",
        "Alugar capacidade de processamento pela internet reduz custos.",
    ]

    catalog = build_reference_catalog(documents)

    assert documents[0] in catalog[1]
    assert documents[1] in catalog[9]
    assert documents[2] in catalog[3]


def test_main_writes_all_model_outputs_and_metrics(tmp_path, monkeypatch):
    input_path = tmp_path / "input.txt"
    input_path.write_text(
        "Profissões inteiras estão sendo reinventadas por programas.\n"
        "Criptografar informações sensíveis impede leituras não autorizadas.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "saida"

    def fake_run_model(model_name, documents, questions):
        return ModelRun(
            model_name=model_name,
            results=tuple(
                QuestionResult(
                    number,
                    question,
                    tuple(
                        Match(document, rank / 10)
                        for rank, document in enumerate(documents, 1)
                    ),
                )
                for number, question in enumerate(questions, 1)
            )
            ,
            indexing_seconds=1.25,
            query_seconds=0.50,
            total_seconds=1.75,
        )

    monkeypatch.setattr(trabalho3, "run_model", fake_run_model)

    exit_code = main(
        ["--input", str(input_path), "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    for model in trabalho3.MODELS:
        assert (output_dir / output_filename(model)).is_file()
    metrics = json.loads(
        (output_dir / "trabalho3-metricas.json").read_text(encoding="utf-8")
    )
    assert set(metrics["models"]) == set(trabalho3.MODELS)
    for model in trabalho3.MODELS:
        assert metrics["models"][model]["performance"] == {
            "indexing_seconds": 1.25,
            "query_seconds": 0.50,
            "total_seconds": 1.75,
        }
