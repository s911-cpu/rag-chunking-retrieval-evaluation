"""
Consolidated End-to-End Pipeline
Evaluating Chunking and Retrieval Parameters for Answer Reliability in RAG-Based AI/ML Research Paper Analysis

Modules:
- Module 1: Dataset & Document Processing (arXiv HTML scraping, cleaning, structuring)
- Module 2: Chunking, Embedding Windows, FAISS Retrieval Experiment
- Module 3: LLM Generation (RAG and No-RAG baselines with Qwen2.5-3B-Instruct)
- Module 4: LLM-as-a-Judge Evaluation, Recall@K, Analysis & Plotting
"""

import ast
import gc
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# Global Configuration & Path Settings
# ==============================================================================
BASE_DIR = os.path.abspath("./rag_evaluation_experiment")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

for directory in [
    DATA_RAW_DIR,
    DATA_METADATA_DIR,
    DATA_PROCESSED_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
]:
    os.makedirs(directory, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


# ==============================================================================
# MODULE 1: Dataset & Document Processing
# ==============================================================================
def get_initial_paper_corpus() -> List[Dict[str, str]]:
    return [
        {
            "paper_id": "P01",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "arxiv_id": "2005.11401",
            "topic": "RAG",
        },
        {
            "paper_id": "P02",
            "title": "Dense Passage Retrieval for Open-Domain Question Answering",
            "arxiv_id": "2004.04906",
            "topic": "Information Retrieval",
        },
        {
            "paper_id": "P03",
            "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
            "arxiv_id": "2312.10997",
            "topic": "RAG Survey",
        },
        {
            "paper_id": "P04",
            "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
            "arxiv_id": "2309.15217",
            "topic": "RAG Evaluation",
        },
        {
            "paper_id": "P05",
            "title": "Lost in the Middle: How Language Models Use Long Contexts",
            "arxiv_id": "2307.03172",
            "topic": "Long Context",
        },
        {
            "paper_id": "P06",
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762",
            "topic": "Transformer",
        },
        {
            "paper_id": "P07",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "arxiv_id": "1810.04805",
            "topic": "Language Model",
        },
        {
            "paper_id": "P08",
            "title": "Language Models are Few-Shot Learners",
            "arxiv_id": "2005.14165",
            "topic": "Large Language Model",
        },
        {
            "paper_id": "P09",
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "arxiv_id": "2106.09685",
            "topic": "LLM Fine-tuning",
        },
        {
            "paper_id": "P10",
            "title": "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation",
            "arxiv_id": "2408.08067",
            "topic": "RAG Evaluation",
        },
    ]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(
        r"\[\s*\d+(?:\s*[\-\–\—]\s*\d+)?(?:\s*,\s*\d+(?:\s*[\-\–\—]\s*\d+)?)*\s*\]",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def extract_sections_from_html(
    html_file: str, paper_id: str
) -> pd.DataFrame:
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "lxml")
    records = []
    headings = soup.find_all(["h2", "h3", "h4"])

    for heading in headings:
        section_title = heading.get_text(" ", strip=True)
        if not section_title:
            continue
        section_text = []
        for element in heading.find_all_next():
            if element.name in ["h2", "h3", "h4"]:
                break
            if element.name in ["p", "li"]:
                txt = element.get_text(" ", strip=True)
                if txt:
                    section_text.append(txt)

        records.append(
            {
                "paper_id": paper_id,
                "heading_level": heading.name,
                "section_title": section_title,
                "text": " ".join(section_text),
            }
        )

    return pd.DataFrame(records)


def run_module_1() -> pd.DataFrame:
    papers = get_initial_paper_corpus()
    metadata_df = pd.DataFrame(papers)
    metadata_path = os.path.join(DATA_METADATA_DIR, "paper_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)

    for _, row in metadata_df.iterrows():
        paper_id, arxiv_id = row["paper_id"], row["arxiv_id"]
        output_file = os.path.join(DATA_RAW_DIR, f"{paper_id}_{arxiv_id}.html")
        if not os.path.exists(output_file):
            url = f"https://arxiv.org/html/{arxiv_id}"
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(response.text)
            except Exception:
                pass
            time.sleep(1)

    all_processed = []
    for _, row in metadata_df.iterrows():
        paper_id, arxiv_id = row["paper_id"], row["arxiv_id"]
        html_file = os.path.join(DATA_RAW_DIR, f"{paper_id}_{arxiv_id}.html")
        if not os.path.exists(html_file):
            continue
        sections = extract_sections_from_html(html_file, paper_id)
        if sections.empty:
            continue
        sections["text"] = sections["text"].apply(clean_text)
        sections = sections[sections["text"].str.strip() != ""].copy()
        sections.reset_index(drop=True, inplace=True)
        sections.to_csv(
            os.path.join(DATA_PROCESSED_DIR, f"{paper_id}_sections_clean.csv"),
            index=False,
            encoding="utf-8",
        )
        all_processed.append(sections)

    module1_dataset = pd.concat(all_processed, ignore_index=True)
    module1_output_path = os.path.join(
        DATA_PROCESSED_DIR, "all_papers_sections_clean.csv"
    )
    module1_dataset.to_csv(module1_output_path, index=False, encoding="utf-8")
    return module1_dataset


# ==============================================================================
# MODULE 2: Chunking, Window Embeddings & Multi-Parameter Retrieval
# ==============================================================================
class EmbeddingRetriever:

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def tokenize_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def create_chunks(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
        paper_id: str = None,
        section: str = None,
        start_chunk_number: int = 1,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")
        tokens = self.tokenize_text(text)
        chunks = []
        start = 0
        chunk_number = start_chunk_number

        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(
                chunk_tokens, skip_special_tokens=True
            )
            chunks.append(
                {
                    "chunk_id": f"C{chunk_number:06d}",
                    "paper_id": paper_id,
                    "section": section,
                    "text": chunk_text,
                }
            )
            start = end - overlap
            chunk_number += 1

        return chunks, chunk_number

    def create_embedding_windows(
        self, chunk: Dict[str, Any], window_size: int = 500
    ) -> List[Dict[str, Any]]:
        tokens = self.tokenize_text(chunk["text"])
        windows = []
        start = 0
        window_number = 1

        while start < len(tokens):
            end = start + window_size
            window_tokens = tokens[start:end]
            window_text = self.tokenizer.decode(
                window_tokens, skip_special_tokens=True
            )
            windows.append(
                {
                    "window_id": f"{chunk['chunk_id']}_W{window_number:03d}",
                    "parent_chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "section": chunk["section"],
                    "text": window_text,
                    "token_ids": window_tokens,
                }
            )
            start = end
            window_number += 1

        return windows

    def create_embeddings(
        self, windows: List[Dict[str, Any]], batch_size: int = 32
    ) -> np.ndarray:
        all_embeddings = []
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            batch_input_ids = []
            for window in batch:
                input_ids = (
                    [self.tokenizer.cls_token_id]
                    + window["token_ids"]
                    + [self.tokenizer.sep_token_id]
                )
                batch_input_ids.append(input_ids)

            max_len = max(len(ids) for ids in batch_input_ids)
            padded_input_ids, attention_masks = [], []
            for ids in batch_input_ids:
                pad_len = max_len - len(ids)
                padded_input_ids.append(
                    ids + [self.tokenizer.pad_token_id] * pad_len
                )
                attention_masks.append([1] * len(ids) + [0] * pad_len)

            input_tensor = torch.tensor(
                padded_input_ids, dtype=torch.long, device=self.device
            )
            mask_tensor = torch.tensor(
                attention_masks, dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor, attention_mask=mask_tensor
                )
                cls_repr = outputs.last_hidden_state[:, 0]
                normalized = F.normalize(cls_repr, p=2, dim=1)
                all_embeddings.extend(normalized.cpu().numpy())

        return np.array(all_embeddings, dtype="float32")

    def create_query_embedding(self, query: str) -> np.ndarray:
        query_instruction = (
            "Represent this sentence for searching relevant passages: "
        )
        query_text = query_instruction + query
        input_ids = self.tokenizer.encode(query_text, add_special_tokens=True)
        input_tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=self.device
        )
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor, attention_mask=attention_mask
            )
            cls_repr = outputs.last_hidden_state[:, 0]
            normalized = F.normalize(cls_repr, p=2, dim=1)

        return normalized.cpu().numpy().astype("float32")

    def build_resources(
        self, documents: pd.DataFrame, chunk_size: int, overlap: int
    ) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]], Dict[str, Any]]:
        all_chunks = []
        next_chunk_number = 1
        for _, row in documents.iterrows():
            chunks, next_chunk_number = self.create_chunks(
                text=row["text"],
                chunk_size=chunk_size,
                overlap=overlap,
                paper_id=row["paper_id"],
                section=row["section_title"],
                start_chunk_number=next_chunk_number,
            )
            all_chunks.extend(chunks)

        parent_chunks = {c["chunk_id"]: c for c in all_chunks}
        all_windows = []
        for chunk in all_chunks:
            all_windows.extend(
                self.create_embedding_windows(chunk, window_size=500)
            )

        embeddings = self.create_embeddings(all_windows, batch_size=32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index, all_windows, parent_chunks

    def retrieve(
        self,
        query: str,
        index: faiss.IndexFlatIP,
        windows: List[Dict[str, Any]],
        parent_chunks: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        q_emb = self.create_query_embedding(query)
        distances, indices = index.search(q_emb, index.ntotal)
        retrieved_chunks = []
        seen_chunks = set()

        for dist, idx in zip(distances[0], indices[0]):
            window = windows[idx]
            p_id = window["parent_chunk_id"]
            if p_id in seen_chunks:
                continue
            seen_chunks.add(p_id)
            p_chunk = parent_chunks[p_id]
            retrieved_chunks.append(
                {
                    "rank": len(retrieved_chunks) + 1,
                    "window_id": window["window_id"],
                    "parent_chunk_id": p_id,
                    "paper_id": p_chunk["paper_id"],
                    "section": p_chunk["section"],
                    "text": p_chunk["text"],
                    "similarity_score": float(dist),
                }
            )
            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks


def run_module_2(
    documents: pd.DataFrame,
    queries: List[Dict[str, str]],
    chunk_sizes: List[int] = [250, 500, 750, 1000, 1500],
    overlaps: List[int] = [0, 50, 100, 200],
    top_k_values: List[int] = [1, 3, 5, 10],
) -> pd.DataFrame:
    retriever = EmbeddingRetriever(device=DEVICE)
    checkpoint_file = os.path.join(
        OUTPUT_DIR, "module_2_retrieval_checkpoint.csv"
    )

    if os.path.exists(checkpoint_file):
        saved = pd.read_csv(checkpoint_file)
        completed = set(saved["config_id"].unique())
    else:
        completed = set()

    config_number = 1
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            config_ids = [
                f"CFG_{config_number + i:03d}"
                for i in range(len(top_k_values))
            ]
            if all(cid in completed for cid in config_ids):
                config_number += len(top_k_values)
                continue

            index, all_windows, parent_chunks = retriever.build_resources(
                documents, chunk_size, overlap
            )
            max_top_k = max(top_k_values)

            for top_k in top_k_values:
                config_id = f"CFG_{config_number:03d}"
                config_number += 1
                if config_id in completed:
                    continue

                config_rows = []
                for q_item in queries:
                    qid, q = q_item["query_id"], q_item["query"]
                    results = retriever.retrieve(
                        query=q,
                        index=index,
                        windows=all_windows,
                        parent_chunks=parent_chunks,
                        top_k=max_top_k,
                    )
                    for res in results[:top_k]:
                        config_rows.append(
                            {
                                "config_id": config_id,
                                "query_id": qid,
                                "query": q,
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "top_k": top_k,
                                "rank": res["rank"],
                                "window_id": res["window_id"],
                                "parent_chunk_id": res["parent_chunk_id"],
                                "paper_id": res["paper_id"],
                                "section": res["section"],
                                "text": res["text"],
                                "similarity_score": res["similarity_score"],
                            }
                        )

                config_df = pd.DataFrame(config_rows)
                if not config_df.empty:
                    config_df.to_csv(
                        checkpoint_file,
                        mode="a",
                        header=not os.path.exists(checkpoint_file),
                        index=False,
                    )
                completed.add(config_id)

    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    retrieval_results = pd.read_csv(checkpoint_file)
    retrieval_results.to_csv(
        os.path.join(OUTPUT_DIR, "module_2_retrieval_results.csv"), index=False
    )
    return retrieval_results


# ==============================================================================
# MODULE 3: LLM Generation (RAG and Baseline No-RAG)
# ==============================================================================
class LLMGenerator:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        self.model.eval()

    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 220,
        max_input_tokens: int = 6000,
    ) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

    def generate_no_rag(self, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful research assistant. Answer the question correctly. "
                    "Do not invent citations or claim to have read documents you have not. "
                    "If you are unsure, say so."
                ),
            },
            {"role": "user", "content": query},
        ]
        return self.generate_from_messages(messages)

    def generate_rag(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful research-paper question-answering assistant. "
                    "Answer using the supplied context. Do not add unsupported claims. "
                    "If the context does not contain enough evidence, explicitly state that "
                    "the provided excerpts are insufficient. Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Retrieved research-paper excerpts:\n{context}\n\n"
                    f"Answer the question using the excerpts above. If the answer is not supported "
                    f"by the excerpts, state that clearly."
                ),
            },
        ]
        return self.generate_from_messages(messages)


def build_context(group: pd.DataFrame) -> str:
    group_sorted = group.sort_values("rank", ascending=True)
    parts = []
    for _, row in group_sorted.iterrows():
        p_id = str(row.get("paper_id", "unknown"))
        sec = str(row.get("section", "unknown"))
        rnk = row.get("rank", "")
        txt = str(row.get("text", "")).strip()
        if txt:
            parts.append(
                f"[Retrieved chunk rank={rnk}; paper_id={p_id}; section={sec}]\n{txt}"
            )
    return "\n\n".join(parts)


def run_module_3(retrieval_results: pd.DataFrame) -> pd.DataFrame:
    generator = LLMGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct", device=DEVICE
    )
    group_cols = ["config_id", "query", "chunk_size", "overlap", "top_k"]
    groups = list(
        retrieval_results.groupby(group_cols, dropna=False, sort=False)
    )

    all_result_rows = []
    no_rag_cache = {}

    for keys, grp in tqdm(groups, desc="Running Generation"):
        config_id, query, chunk_size, overlap, top_k = keys
        grp_sorted = grp.sort_values("rank", ascending=True)
        context = build_context(grp_sorted)

        try:
            if query not in no_rag_cache:
                no_rag_cache[query] = generator.generate_no_rag(query)
            no_rag_answer = no_rag_cache[query]
            rag_answer = generator.generate_rag(query, context)
            status, error_msg = "success", ""
        except Exception as e:
            rag_answer = ""
            no_rag_answer = no_rag_cache.get(query, "")
            status, error_msg = "error", repr(e)

        p_ids = [
            str(x) for x in grp_sorted["paper_id"].dropna().unique().tolist()
        ]
        c_ids = (
            grp_sorted["parent_chunk_id"].dropna().astype(str).tolist()
            if "parent_chunk_id" in grp_sorted.columns
            else []
        )
        w_ids = (
            grp_sorted["window_id"].dropna().astype(str).tolist()
            if "window_id" in grp_sorted.columns
            else []
        )
        sim_scores = grp_sorted["similarity_score"].dropna().tolist()

        all_result_rows.append(
            {
                "config_id": config_id,
                "query": query,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k": top_k,
                "num_retrieved_chunks": len(grp_sorted),
                "retrieved_paper_ids": json.dumps(p_ids, ensure_ascii=False),
                "retrieved_parent_chunk_ids": json.dumps(
                    c_ids, ensure_ascii=False
                ),
                "retrieved_window_ids": json.dumps(w_ids, ensure_ascii=False),
                "similarity_scores": json.dumps(sim_scores),
                "retrieved_context": context,
                "rag_answer": rag_answer,
                "no_rag_answer": no_rag_answer,
                "model_name": "Qwen/Qwen2.5-3B-Instruct",
                "max_new_tokens": 220,
                "generation_mode": "greedy",
                "generation_status": status,
                "error_message": error_msg,
            }
        )

    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generation_results = pd.DataFrame(all_result_rows)
    output_path = os.path.join(
        OUTPUT_DIR, "module_3_generation_results.csv"
    )
    generation_results.to_csv(output_path, index=False)
    return generation_results


# ==============================================================================
# MODULE 4: Evaluation, Judging, Comparison, and Visualization
# ==============================================================================
JUDGE_RUBRIC_SYSTEM = """You are an evaluator for an experiment studying retrieval-augmented generation (RAG).
Evaluate answer quality and hallucination as two independent scores.
ANSWER QUALITY:
2 = Correct and sufficiently answers the question. Different wording is acceptable.
1 = Partly correct, incomplete, vague, or contains a meaningful error, but still addresses part of the question.
0 = Incorrect, irrelevant, answers a different question, or fails to provide a meaningful answer.
HALLUCINATION:
1 = At least one clear factual claim is unsupported by or contradicts the retrieved context.
0 = No clear unsupported or contradicted factual claim is present.
Return exactly one JSON object:
{"answer_quality": integer 0, 1, or 2, "hallucination": integer 0 or 1, "reason": "short explanation"}"""


def parse_judge_output(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return {
            "answer_quality": pd.NA,
            "hallucination": pd.NA,
            "judge_reason": "Invalid JSON",
            "judge_parse_status": "parse_error",
        }
    try:
        obj = json.loads(match.group(0))
        q, h, r = (
            obj.get("answer_quality"),
            obj.get("hallucination"),
            obj.get("reason", ""),
        )
        if type(q) is int and q in (0, 1, 2) and type(h) is int and h in (0, 1):
            return {
                "answer_quality": q,
                "hallucination": h,
                "judge_reason": str(r)[:1000],
                "judge_parse_status": "ok",
            }
    except Exception:
        pass
    return {
        "answer_quality": pd.NA,
        "hallucination": pd.NA,
        "judge_reason": "Schema validation failure",
        "judge_parse_status": "schema_error",
    }


def run_judge_pipeline(
    work_df: pd.DataFrame,
    output_path: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    batch_size: int = 4,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    results = work_df.copy().reset_index(drop=True)
    for col in [
        "answer_quality",
        "hallucination",
        "judge_reason",
        "judge_parse_status",
    ]:
        if col not in results.columns:
            results[col] = pd.NA

    pending = [
        idx
        for idx, row in results.iterrows()
        if pd.isna(row["answer_quality"]) or pd.isna(row["hallucination"])
    ]

    for start in tqdm(
        range(0, len(pending), batch_size), desc="Judging Answers"
    ):
        indices = pending[start : start + batch_size]
        prompts = []
        for idx in indices:
            row = results.loc[idx]
            context = str(row.get("retrieved_context", ""))[:5000]
            messages = [
                {"role": "system", "content": JUDGE_RUBRIC_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{row['query']}\n\n"
                        f"REFERENCE ANSWER:\n{row.get('expected_answer', '')}\n\n"
                        f"RETRIEVED CONTEXT:\n{context}\n\n"
                        f"RAG ANSWER:\n{row.get('rag_answer', '')}\n\n"
                        f"Evaluate this answer using the rubric."
                    ),
                },
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=6000,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        width = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(
            outputs[:, width:], skip_special_tokens=True
        )

        for idx, raw in zip(indices, decoded):
            parsed = parse_judge_output(raw)