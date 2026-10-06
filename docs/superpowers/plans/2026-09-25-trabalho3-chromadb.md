# Trabalho 3 ChromaDB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar em uma unica pasta o codigo Python, os dois arquivos de resultados reais e o relatorio comparativo solicitados no Trabalho 3.

**Architecture:** Um modulo Python concentra constantes, leitura do corpus, indexacao/consulta via ChromaDB e serializacao. Funcoes puras isolam formatacao e metricas para permitir testes sem download de modelos; a execucao real injeta a funcao de embeddings do ChromaDB para cada modelo. Um segundo script interno gera o PDF a partir dos resultados verificados, mantendo os artefatos de envio simples.

**Tech Stack:** Python 3.11+, ChromaDB, sentence-transformers, pytest, ReportLab, Poppler.

**Spec:** `docs/superpowers/specs/2026-09-25-trabalho3-chromadb-design.md`

## Global Constraints

- Usar exatamente os modelos `all-MiniLM-L6-v2` e `all-mpnet-base-v2`.
- Consultar as 10 perguntas do enunciado com cinco resultados por modelo e pergunta.
- Processar integralmente o corpus real de 1.000 frases UTF-8 e registrar a divergencia em relacao as 10.000 frases citadas no PDF.
- Manter busca vetorial direta, sem expansao de consulta ou reranqueamento.
- Colocar todos os arquivos finais em `entrega-trabalho3/`.
- Os dois arquivos `.txt` entregues devem ser gerados pela execucao real do codigo.
- Nao comparar diretamente valores absolutos de distancia entre modelos diferentes.

## Review Focus

- Arquivo UTF-8 com linhas vazias: ignorar somente vazias e preservar texto/ordem das demais.
- Corpus vazio: falhar antes de inicializar ChromaDB ou baixar modelos.
- Corpus com menos de cinco frases: retornar todas sem solicitar mais resultados do que documentos existentes.
- Resposta ChromaDB incompleta ou desalinhada: falhar com mensagem clara, sem escrever arquivo final parcial.
- Execucao repetida: substituir os arquivos finais de forma atomica e produzir conteudo deterministico para os mesmos embeddings.

---

### Task 1: Leitura, validacao e formato dos resultados

**Files:**
- Create: `tests/test_trabalho3.py`
- Create: `entrega-trabalho3/trabalho3.py`

**Interfaces:**
- Produces: `read_corpus(path: Path) -> list[str]`
- Produces: `stable_ids(count: int) -> list[str]`
- Produces: `format_output(model_name: str, corpus_size: int, results: list[QuestionResult]) -> str`
- Produces: dataclasses `Match` and `QuestionResult`

- [ ] **Step 1: Write failing tests for corpus parsing and stable IDs**

```python
def test_read_corpus_preserves_utf8_order_and_ignores_blank_lines(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("saude digital\n\n  \ncomputacao em nuvem\n", encoding="utf-8")
    assert read_corpus(source) == ["saude digital", "computacao em nuvem"]

def test_read_corpus_rejects_empty_input(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="nenhuma frase"):
        read_corpus(source)

def test_stable_ids_are_unique_and_zero_padded():
    assert stable_ids(3) == ["frase-000001", "frase-000002", "frase-000003"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_trabalho3.py -v`
Expected: collection/import failure because `entrega-trabalho3/trabalho3.py` does not exist.

- [ ] **Step 3: Implement dataclasses, constants, parser and IDs**

```python
@dataclass(frozen=True)
class Match:
    document: str
    distance: float

@dataclass(frozen=True)
class QuestionResult:
    number: int
    question: str
    matches: tuple[Match, ...]

def read_corpus(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    documents = [line for line in lines if line]
    if not documents:
        raise ValueError("O arquivo de entrada nao contem nenhuma frase.")
    return documents

def stable_ids(count: int) -> list[str]:
    return [f"frase-{index:06d}" for index in range(1, count + 1)]
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_trabalho3.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write failing format test**

```python
def test_format_output_contains_ten_questions_and_five_ranked_matches():
    results = sample_results(question_count=10, matches_per_question=5)
    text = format_output("all-MiniLM-L6-v2", 1000, results)
    assert "Modelo: all-MiniLM-L6-v2" in text
    assert "Frases indexadas: 1000" in text
    assert text.count("PERGUNTA ") == 10
    assert text.count("Distancia:") == 50
    assert "1. Como a inteligencia artificial" in text
```

- [ ] **Step 6: Run format test and verify RED**

Run: `python -m pytest tests/test_trabalho3.py::test_format_output_contains_ten_questions_and_five_ranked_matches -v`
Expected: FAIL because `format_output` is missing.

- [ ] **Step 7: Implement deterministic UTF-8 output formatting**

```python
def format_output(model_name: str, corpus_size: int, results: list[QuestionResult]) -> str:
    lines = [
        "TRABALHO 3 - BUSCA SEMANTICA COM CHROMADB",
        f"Modelo: {model_name}",
        f"Frases indexadas: {corpus_size}",
        "Resultados por pergunta: 5",
        "Metrica: distancia cosseno (menor e melhor)",
        "",
    ]
    for result in results:
        lines.extend([f"PERGUNTA {result.number}", f"{result.number}. {result.question}"])
        for rank, match in enumerate(result.matches, 1):
            lines.append(f"  {rank}) Distancia: {match.distance:.6f} | {match.document}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 8: Run entire test file and verify GREEN**

Run: `python -m pytest tests/test_trabalho3.py -v`
Expected: all tests pass.

### Task 2: Busca vetorial real com ChromaDB

**Files:**
- Modify: `tests/test_trabalho3.py`
- Modify: `entrega-trabalho3/trabalho3.py`
- Create: `entrega-trabalho3/requirements.txt`

**Interfaces:**
- Consumes: `Match`, `QuestionResult`, `stable_ids()` from Task 1.
- Produces: `query_collection(collection, questions: Sequence[str], n_results: int = 5) -> list[QuestionResult]`
- Produces: `run_model(model_name: str, documents: list[str], questions: Sequence[str]) -> list[QuestionResult]`
- Produces: `atomic_write_text(path: Path, content: str) -> None`

- [ ] **Step 1: Write failing tests for querying and short corpora**

```python
def test_query_collection_maps_documents_and_distances_in_rank_order():
    collection = FakeCollection({
        "documents": [["frase A", "frase B"]],
        "distances": [[0.10, 0.25]],
    })
    result = query_collection(collection, ["pergunta"], n_results=2)
    assert result[0].matches == (Match("frase A", 0.10), Match("frase B", 0.25))
    assert collection.requested_n_results == 2

def test_run_model_caps_n_results_to_corpus_size(monkeypatch):
    factory = FakeChromaFactory()
    monkeypatch.setattr(trabalho3, "create_collection", factory)
    results = run_model("all-MiniLM-L6-v2", ["A", "B"], ["pergunta"])
    assert len(results[0].matches) == 2
    assert factory.collection.requested_n_results == 2
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_trabalho3.py -k "query_collection or caps_n_results" -v`
Expected: FAIL because vector query functions are missing.

- [ ] **Step 3: Implement Chroma collection creation and query mapping**

```python
def create_collection(model_name: str):
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    client = chromadb.Client()
    embedding = SentenceTransformerEmbeddingFunction(model_name=model_name)
    return client.create_collection(
        name="busca-" + hashlib.sha1(model_name.encode()).hexdigest()[:12],
        embedding_function=embedding,
        metadata={"hnsw:space": "cosine"},
    )

def query_collection(collection, questions: Sequence[str], n_results: int = 5) -> list[QuestionResult]:
    raw = collection.query(query_texts=list(questions), n_results=n_results,
                           include=["documents", "distances"])
    return [QuestionResult(i, question, tuple(Match(doc, float(distance))
            for doc, distance in zip(raw["documents"][i-1], raw["distances"][i-1], strict=True)))
            for i, question in enumerate(questions, 1)]

def run_model(model_name: str, documents: list[str], questions: Sequence[str]) -> list[QuestionResult]:
    collection = create_collection(model_name)
    collection.add(ids=stable_ids(len(documents)), documents=documents)
    return query_collection(collection, questions, min(5, len(documents)))
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_trabalho3.py -k "query_collection or caps_n_results" -v`
Expected: tests pass.

- [ ] **Step 5: Write failing atomic-output test**

```python
def test_atomic_write_replaces_destination_without_temp_file(tmp_path):
    target = tmp_path / "saida.txt"
    target.write_text("antigo", encoding="utf-8")
    atomic_write_text(target, "novo\n")
    assert target.read_text(encoding="utf-8") == "novo\n"
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 6: Run atomic test and verify RED**

Run: `python -m pytest tests/test_trabalho3.py::test_atomic_write_replaces_destination_without_temp_file -v`
Expected: FAIL because `atomic_write_text` is missing.

- [ ] **Step 7: Implement atomic output and CLI orchestration**

```python
def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    documents = read_corpus(args.input)
    all_results = {}
    for model_name in MODELS:
        results = run_model(model_name, documents, QUESTIONS)
        destination = args.output_dir / f"trabalho3-output-{model_name}.txt"
        atomic_write_text(destination, format_output(model_name, len(documents), results))
        all_results[model_name] = results
    write_metrics(args.output_dir / "trabalho3-metricas.json", all_results, documents)
    return 0
```

- [ ] **Step 8: Add dependency manifest**

```text
chromadb>=1.0,<2
sentence-transformers>=3.0,<6
pytest>=8,<10
reportlab>=4,<5
```

- [ ] **Step 9: Run full unit suite**

Run: `python -m pytest -v`
Expected: all tests pass with no network access.

### Task 3: Metricas comparativas e integracao real

**Files:**
- Modify: `tests/test_trabalho3.py`
- Modify: `entrega-trabalho3/trabalho3.py`
- Create: `entrega-trabalho3/trabalho3-input.txt`
- Generate: `entrega-trabalho3/trabalho3-output-all-MiniLM-L6-v2.txt`
- Generate: `entrega-trabalho3/trabalho3-output-all-mpnet-base-v2.txt`
- Generate: `entrega-trabalho3/trabalho3-metricas.json`

**Interfaces:**
- Consumes: result dataclasses and both model result lists.
- Produces: `compare_results(results_by_model: Mapping[str, list[QuestionResult]], relevant_documents: Mapping[int, set[str]]) -> dict`
- Produces: JSON schema with per-model `recall_at_5`, `mrr`, per-question relevance, and cross-model overlap.

- [ ] **Step 1: Write failing metric tests with hand-calculated examples**

```python
def test_compare_results_calculates_recall_mrr_and_overlap():
    model_a = [QuestionResult(1, "Q", (Match("R1", .1), Match("X", .2)))]
    model_b = [QuestionResult(1, "Q", (Match("X", .1), Match("R1", .2)))]
    metrics = compare_results({"A": model_a, "B": model_b}, {1: {"R1", "R2"}})
    assert metrics["models"]["A"]["recall_at_5"] == pytest.approx(0.5)
    assert metrics["models"]["A"]["mrr"] == pytest.approx(1.0)
    assert metrics["models"]["B"]["mrr"] == pytest.approx(0.5)
    assert metrics["overlap"]["1"] == 2
```

- [ ] **Step 2: Run metric test and verify RED**

Run: `python -m pytest tests/test_trabalho3.py::test_compare_results_calculates_recall_mrr_and_overlap -v`
Expected: FAIL because `compare_results` is missing.

- [ ] **Step 3: Implement metrics and reference-set builder**

```python
def score_question(result: QuestionResult, relevant: set[str]) -> dict[str, float | int]:
    retrieved = [match.document for match in result.matches]
    relevant_ranks = [rank for rank, document in enumerate(retrieved, 1) if document in relevant]
    return {
        "relevant_retrieved": len(relevant_ranks),
        "reference_total": len(relevant),
        "recall_at_5": len(relevant_ranks) / len(relevant) if relevant else 0.0,
        "reciprocal_rank": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
    }
```

The reference catalog will contain the exact corpus sentences that unambiguously belong to each of the 10 topics. It will be stored as a constant mapping with the question number as key, and every entry will be validated against the loaded corpus before scoring.

- [ ] **Step 4: Run metric tests and full suite**

Run: `python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Copy the original corpus into the delivery folder**

Run: `Copy-Item -LiteralPath 'trabalho3-input.txt' -Destination 'entrega-trabalho3\trabalho3-input.txt'`
Expected: SHA-256 of source and copy are identical.

- [ ] **Step 6: Install dependencies and run the real pipeline**

Run: `python -m pip install -r entrega-trabalho3/requirements.txt`
Run: `python entrega-trabalho3/trabalho3.py --input entrega-trabalho3/trabalho3-input.txt --output-dir entrega-trabalho3`
Expected: both models download/load, 1.000 documents are indexed twice, and three result artifacts are generated.

- [ ] **Step 7: Validate generated outputs structurally**

Run: `python -m pytest tests/test_integration_outputs.py -v`
Expected: each output names the correct model, contains 10 questions and 50 distance lines; every returned sentence occurs verbatim in the corpus; JSON parses and contains both models.

### Task 4: Relatorio, documentacao e verificacao final

**Files:**
- Create: `tests/test_integration_outputs.py`
- Create: `tools/build_report.py`
- Create: `entrega-trabalho3/README.md`
- Create: `entrega-trabalho3/relatorio-trabalho3.md`
- Generate: `entrega-trabalho3/relatorio-trabalho3.pdf`

**Interfaces:**
- Consumes: both output text files and `trabalho3-metricas.json`.
- Produces: final editable report and visually verified PDF.

- [ ] **Step 1: Write output integrity tests before report generation**

```python
@pytest.mark.parametrize("model", MODELS)
def test_real_output_has_fifty_verbatim_matches(model, delivery_dir, corpus):
    text = (delivery_dir / f"trabalho3-output-{model}.txt").read_text(encoding="utf-8")
    assert text.count("PERGUNTA ") == 10
    lines = [line for line in text.splitlines() if "Distancia:" in line]
    assert len(lines) == 50
    for line in lines:
        document = line.split(" | ", 1)[1]
        assert document in corpus
```

- [ ] **Step 2: Run integrity tests and fix only generator defects**

Run: `python -m pytest tests/test_integration_outputs.py -v`
Expected: PASS; any failure requires a failing unit regression test before code correction.

- [ ] **Step 3: Write README with exact reproduction commands**

Document Python requirement, virtual environment commands for PowerShell, dependency installation, the single execution command, generated filenames, expected runtime/download, and the verified 1.000-versus-10.000 discrepancy.

- [ ] **Step 4: Draft report from verified metrics**

Write `relatorio-trabalho3.md` with the following complete sections: Resumo; Objetivo; Base de dados; Metodologia; Resultados por pergunta; Comparacao quantitativa; Analise qualitativa; Por que os modelos diferem; Vencedor; Limitacoes; Conclusao; Referencias. Populate every table and claim from the actual JSON and output files.

- [ ] **Step 5: Mark PDF authoring operation once**

Run: `node container_tools/mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format pdf`
Expected: command exits successfully exactly once before PDF creation.

- [ ] **Step 6: Generate the PDF with ReportLab**

Run: `python tools/build_report.py`
Expected: `entrega-trabalho3/relatorio-trabalho3.pdf` is generated with cover, page numbers, styled headings and readable tables.

- [ ] **Step 7: Render and inspect every PDF page**

Run: `pdftoppm -png -r 150 entrega-trabalho3/relatorio-trabalho3.pdf tmp/pdfs/relatorio-trabalho3`
Expected: every page has complete text, consistent margins, no overlap, no clipping and correct Portuguese glyphs. Adjust builder and regenerate until clean.

- [ ] **Step 8: Run final verification suite**

Run: `python -m pytest -v`
Expected: all unit and integration tests pass with no warnings or failures.

- [ ] **Step 9: Verify delivery inventory and hashes**

Run: `Get-ChildItem -File entrega-trabalho3 | Select-Object Name,Length; Get-FileHash entrega-trabalho3/trabalho3-input.txt, trabalho3-input.txt`
Expected: code, input, two nonempty outputs, metrics, README, editable report and PDF are present; input hashes match.

