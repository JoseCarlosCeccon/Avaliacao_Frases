"""Trabalho 3: busca semântica com ChromaDB e Sentence Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

# O transporte Xet pode falhar na reconstrução de arquivos grandes no Windows.
# O cliente oficial do Hugging Face respeita esta variável e usa HTTP comum.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


MODELS = (
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-m3",
)
MODEL_FILE_LABELS = {"BAAI/bge-m3": "BGE-M3"}
ADD_BATCH_SIZE = 5000

QUESTIONS = (
    "Como a inteligência artificial está mudando o mercado de trabalho?",
    "Quais são os riscos de segurança relacionados a novas tecnologias?",
    "De que forma a computação em nuvem beneficia as empresas?",
    "Como o big data ajuda na tomada de decisões?",
    "Qual o impacto da automação nos processos industriais?",
    "Como a tecnologia está sendo usada na área da saúde?",
    "Quais tecnologias estão relacionadas à sustentabilidade?",
    "Como o trabalho remoto foi impactado pela tecnologia?",
    "De que forma a cibersegurança protege dados das empresas?",
    "Como a inovação tecnológica influencia startups e novos negócios?",
)

# Fragmentos de frases inequivocamente temáticas. Este catálogo é usado
# somente na avaliação posterior; não interfere na busca nem no ranking.
REFERENCE_FRAGMENTS: dict[int, tuple[str, ...]] = {
    1: (
        "profissões inteiras",
        "empregos que exigiam apenas repetição",
        "vagas de emprego",
        "redefinindo cargos",
        "funções administrativas rotineiras",
    ),
    2: (
        "exploração de vulnerabilidades",
        "programas maliciosos",
        "tentativas de fraude",
        "aplicativos falsos",
        "redes públicas de conexão",
        "vazamentos de informações pessoais",
    ),
    3: (
        "capacidade de processamento pela internet",
        "arquivos em servidores remotos",
        "infraestrutura fica hospedada fora",
        "armazenamento sob demanda",
    ),
    4: (
        "volumes gigantescos de informação",
        "tendências escondidas em conjuntos enormes",
        "milhões de usuários simultaneamente",
        "históricos extensos de compras",
    ),
    5: (
        "tarefas repetitivas em fábricas",
        "trabalho manual agora são concluídos",
        "equipamentos autônomos aceleraram a produção",
        "dispositivos automatizados reduziu o desperdício",
        "repetir movimentos exatos",
    ),
    6: (
        "atendimentos médicos realizados por chamada",
        "registros clínicos digitais",
        "prontuários eletrônicos",
        "consultas médicas à distância",
        "diagnósticos por imagem",
        "detectar tumores",
    ),
    7: (
        "reciclagem de componentes eletrônicos",
        "energia limpa gerada",
        "energia do sol",
        "recursos naturais",
        "vento para gerar eletricidade",
        "economizar água na agricultura",
    ),
    8: (
        "funcionários que trabalham de qualquer lugar",
        "flexibilidade de horário conquistada fora",
        "escritório passou a ser em casa",
        "trabalhar de casa",
        "equipes espalhadas por diferentes cidades",
    ),
    9: (
        "monitorar acessos suspeitos",
        "restringir o acesso de funcionários",
        "criptografar informações sensíveis",
        "autenticação em duas etapas",
    ),
    10: (
        "pequenas empresas de tecnologia",
        "empreendedores criam soluções digitais",
        "negócios recém-criados",
        "ideias que começam em garagens",
    ),
}


@dataclass(frozen=True)
class Match:
    """Uma frase recuperada e sua distância em relação à consulta."""

    document: str
    distance: float


@dataclass(frozen=True)
class QuestionResult:
    """Resultados ordenados de uma pergunta."""

    number: int
    question: str
    matches: tuple[Match, ...]


@dataclass(frozen=True)
class ModelRun:
    """Resultado e tempos observados durante a execução de um modelo."""

    model_name: str
    results: tuple[QuestionResult, ...]
    indexing_seconds: float
    query_seconds: float
    total_seconds: float


def output_filename(model_name: str) -> str:
    """Retorna um nome de arquivo portátil e legível para o modelo."""

    label = MODEL_FILE_LABELS.get(model_name, model_name.replace("/", "-"))
    return f"trabalho3-output-{label}.txt"


def read_corpus(path: Path) -> list[str]:
    """Lê frases UTF-8, descartando somente linhas vazias."""

    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
    ]
    documents = [line for line in lines if line]
    if not documents:
        raise ValueError("O arquivo de entrada não contém nenhuma frase.")
    return documents


def stable_ids(count: int) -> list[str]:
    """Gera identificadores reproduzíveis e únicos para o ChromaDB."""

    return [f"frase-{index:06d}" for index in range(1, count + 1)]


def format_output(
    model_name: str,
    corpus_size: int,
    results: Sequence[QuestionResult],
) -> str:
    """Formata resultados no arquivo textual solicitado."""

    result_count = max((len(result.matches) for result in results), default=0)
    lines = [
        "TRABALHO 3 - BUSCA SEMÂNTICA COM CHROMADB",
        f"Modelo: {model_name}",
        f"Frases indexadas: {corpus_size}",
        f"Resultados por pergunta: {result_count}",
        "Métrica: distância cosseno (menor é melhor)",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"PERGUNTA {result.number}",
                f"{result.number}. {result.question}",
            ]
        )
        for rank, match in enumerate(result.matches, 1):
            lines.append(
                f"  {rank}) Distância: {match.distance:.6f} | {match.document}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def create_collection(model_name: str):
    """Cria uma coleção ChromaDB efêmera com distância cosseno."""

    import chromadb
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )

    client = chromadb.EphemeralClient()
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    name_hash = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:12]
    return client.create_collection(
        name=f"busca-{name_hash}",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def query_collection(
    collection,
    questions: Sequence[str],
    n_results: int = 5,
) -> list[QuestionResult]:
    """Consulta uma coleção e converte a resposta em objetos tipados."""

    raw = collection.query(
        query_texts=list(questions),
        n_results=n_results,
        include=["documents", "distances"],
    )
    documents_by_question = raw.get("documents")
    distances_by_question = raw.get("distances")
    if (
        documents_by_question is None
        or distances_by_question is None
        or len(documents_by_question) != len(questions)
        or len(distances_by_question) != len(questions)
    ):
        raise ValueError("Resposta inconsistente recebida do ChromaDB.")

    results: list[QuestionResult] = []
    for number, question in enumerate(questions, 1):
        documents = documents_by_question[number - 1]
        distances = distances_by_question[number - 1]
        if len(documents) != len(distances):
            raise ValueError("Resposta inconsistente recebida do ChromaDB.")
        matches = tuple(
            Match(document=document, distance=float(distance))
            for document, distance in zip(documents, distances, strict=True)
        )
        results.append(QuestionResult(number, question, matches))
    return results


def run_model(
    model_name: str,
    documents: list[str],
    questions: Sequence[str] = QUESTIONS,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> ModelRun:
    """Indexa, consulta e mede as duas etapas para um modelo."""

    if not documents:
        raise ValueError("O corpus deve conter ao menos uma frase.")
    started = clock()
    collection = create_collection(model_name)
    ids = stable_ids(len(documents))
    for start in range(0, len(documents), ADD_BATCH_SIZE):
        end = start + ADD_BATCH_SIZE
        collection.add(ids=ids[start:end], documents=documents[start:end])
    indexed = clock()
    results = query_collection(collection, questions, min(5, len(documents)))
    finished = clock()
    return ModelRun(
        model_name=model_name,
        results=tuple(results),
        indexing_seconds=indexed - started,
        query_seconds=finished - indexed,
        total_seconds=finished - started,
    )


def atomic_write_text(path: Path, content: str) -> None:
    """Substitui um arquivo apenas depois de concluir a escrita UTF-8."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_reference_catalog(documents: Sequence[str]) -> dict[int, set[str]]:
    """Seleciona frases de referência por fragmentos temáticos explícitos."""

    catalog: dict[int, set[str]] = {number: set() for number in range(1, 11)}
    for document in documents:
        normalized = document.casefold()
        for number, fragments in REFERENCE_FRAGMENTS.items():
            if any(fragment.casefold() in normalized for fragment in fragments):
                catalog[number].add(document)
    return catalog


def _score_question(
    result: QuestionResult, relevant: set[str]
) -> dict[str, float | int]:
    retrieved = [match.document for match in result.matches]
    relevant_ranks = [
        rank
        for rank, document in enumerate(retrieved, 1)
        if document in relevant
    ]
    return {
        "relevant_retrieved": len(relevant_ranks),
        "reference_total": len(relevant),
        "recall_at_5": len(relevant_ranks) / len(relevant) if relevant else 0.0,
        "reciprocal_rank": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
    }


def compare_results(
    results_by_model: Mapping[str, Sequence[QuestionResult]],
    relevant_documents: Mapping[int, set[str]],
) -> dict:
    """Calcula Recall@5, MRR e sobreposição dos rankings."""

    model_metrics: dict[str, dict] = {}
    indexed_results: dict[str, dict[int, QuestionResult]] = {}
    for model_name, results in results_by_model.items():
        by_number = {result.number: result for result in results}
        indexed_results[model_name] = by_number
        per_question: dict[str, dict] = {}
        for number, result in sorted(by_number.items()):
            per_question[str(number)] = _score_question(
                result, set(relevant_documents.get(number, set()))
            )
        recalls = [score["recall_at_5"] for score in per_question.values()]
        reciprocal_ranks = [
            score["reciprocal_rank"] for score in per_question.values()
        ]
        model_metrics[model_name] = {
            "recall_at_5": statistics.fmean(recalls) if recalls else 0.0,
            "mrr": statistics.fmean(reciprocal_ranks)
            if reciprocal_ranks
            else 0.0,
            "per_question": per_question,
        }

    overlap: dict[str, int] = {}
    model_names = list(results_by_model)
    if len(model_names) >= 2:
        first, second = model_names[:2]
        common_questions = set(indexed_results[first]) & set(indexed_results[second])
        for number in sorted(common_questions):
            first_documents = {
                match.document for match in indexed_results[first][number].matches
            }
            second_documents = {
                match.document for match in indexed_results[second][number].matches
            }
            overlap[str(number)] = len(first_documents & second_documents)

    return {"models": model_metrics, "overlap": overlap}


def write_metrics(path: Path, metrics: Mapping) -> None:
    """Grava métricas como JSON UTF-8 legível e determinístico."""

    atomic_write_text(
        path,
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o Trabalho 3 com ChromaDB e quatro modelos."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("trabalho3-input.txt"),
        help="Arquivo UTF-8 com uma frase por linha.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Diretório dos arquivos de saída.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    documents = read_corpus(args.input)
    unique_count = len(set(documents))
    print(f"Corpus: {len(documents)} frases ({unique_count} únicas).")

    results_by_model: dict[str, Sequence[QuestionResult]] = {}
    performance_by_model: dict[str, dict[str, float]] = {}
    for model_name in MODELS:
        print(f"Processando {model_name}...")
        run = run_model(model_name, documents, QUESTIONS)
        output_path = args.output_dir / output_filename(model_name)
        atomic_write_text(
            output_path,
            format_output(model_name, len(documents), run.results),
        )
        results_by_model[model_name] = run.results
        performance_by_model[model_name] = {
            "indexing_seconds": run.indexing_seconds,
            "query_seconds": run.query_seconds,
            "total_seconds": run.total_seconds,
        }
        print(f"Arquivo gerado: {output_path}")
        print(
            "Tempos: "
            f"indexação={run.indexing_seconds:.3f}s, "
            f"consultas={run.query_seconds:.3f}s, "
            f"total={run.total_seconds:.3f}s"
        )

    references = build_reference_catalog(documents)
    metrics = compare_results(results_by_model, references)
    for model_name, performance in performance_by_model.items():
        metrics["models"][model_name]["performance"] = performance
    metrics["corpus"] = {
        "total": len(documents),
        "unique": unique_count,
        "reference_counts": {
            str(number): len(items) for number, items in references.items()
        },
    }
    metrics_path = args.output_dir / "trabalho3-metricas.json"
    write_metrics(metrics_path, metrics)
    print(f"Métricas geradas: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
