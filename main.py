# Importing the dependencies.

import gradio as gr # for user interface
import chromadb # to be used as vector database
import pandas as pd # for handling the CSV data
import json
import re
from typing import List, Dict, Tuple
import torch # for model to be run on cpu or gpu
from sentence_transformers import SentenceTransformer # for embedding model
from transformers import pipeline # for FLAN-T5 model that extracts claims
import os
import warnings # to ignore warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------------------------------------------

# Coding main PIB fact checking class.
class PIBFactChecker:
    def __init__(self, default_csv: str = "pib_facts.csv"):
        print(" Initializing PIB Fact Checker...")

        # Load embedding model
        print("Loading embedding model: all-MiniLM-L6-v2...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # ChromaDB embedding function wrapper
        from chromadb.api.types import EmbeddingFunction
        
        class CustomEmbeddingFunction(EmbeddingFunction):
            def __init__(self, model):
                self.model = model                    
            def __call__(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                return self.model.encode(texts).tolist()
        
        # Pass the model to the embedding function
        self.embedding_fn = CustomEmbeddingFunction(self.embedding_model)

        #   Initialize ChromaDB client and collection with embedding function
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="pib_facts",
            embedding_function=self.embedding_fn   # ← Now it has .model
        )

        # Load default CSV into vector database
        if os.path.exists(default_csv):
            print(f"Found {default_csv} → Loading into vector database...")
            status = self.load_csv_to_vectordb(default_csv)
            print(status)
        else:
            print(f"This {default_csv} not found! Place your CSV in the same folder.")

        # Load FLAN-T5 model for claim extraction
        print("Loading FLAN-T5 for claim extraction...")
        try:
            self.claim_extractor = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            self.claim_extractor = None
            print("FLAN-T5 not available → using rule-based extraction")

        self.vague_threshold = 0.35


    # Load CSV into Vector Database
    def load_csv_to_vectordb(self, csv_path: str, text_column: str = "statement"):
        try:
            df = pd.read_csv(csv_path)
            if text_column not in df.columns:
                return f"Error: '{text_column}' column not found!"

            documents, metadatas, ids = [], [], []
            for idx, row in df.iterrows():
                text = str(row[text_column])
                chunks = self.chunk_text(text)
                for c_idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata = {"source_row": idx, "chunk_id": c_idx}
                    for col in df.columns:
                        if col != text_column:
                            metadata[col] = str(row[col])
                    metadatas.append(metadata)
                    ids.append(f"row{idx}_chunk{c_idx}")

            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            return f"✅ Loaded {len(documents)} chunks from {len(df)} PIB statements."
        except Exception as e:
            return f"Error: {str(e)}"


    # Chunking text into small parts
    def chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) < chunk_size:
                current += " " + s
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = s
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [text]


    # Extract claims from input text
    def extract_claims(self, text: str) -> List[str]:
        candidates = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                     if len(s.strip()) > 20 and not s.strip().endswith('?')]

        if self.claim_extractor and len(text) < 1000:
            try:
                result = self.claim_extractor(
                    f"Extract all verifiable factual claims:\n\n{text}",
                    max_new_tokens=256,
                    do_sample=False
                )
                extracted = result[0]['generated_text']
                model_claims = [c.strip() for c in extracted.split('\n') if len(c.strip()) > 20]
                if model_claims:
                    candidates = model_claims
            except:
                pass
        return candidates if candidates else [text]


    # vagueness detection
    def is_vague_claim(self, claim: str) -> Tuple[bool, float]:
        vague_words = r'\bmany\b|\bsome\b|\boften\b|\busually\b|\bmight\b|\bcould\b|\bmay\b|\bpossibly\b|\bperhaps\b'
        vague_count = len(re.findall(vague_words, claim.lower()))
        specific_patterns = r'\d{4}|\d+%|\d+|[A-Z][a-z]+ [A-Z][a-z]+'
        specific_count = len(re.findall(specific_patterns, claim))
        vagueness_score = vague_count / max(1, len(claim.split()))
        specificity_score = specific_count / max(1, len(claim.split()))
        final_score = vagueness_score - specificity_score
        return final_score > self.vague_threshold, final_score


    # retrieve similar facts from vector database
    def retrieve_similar_facts(self, claim: str, top_k: int = 5) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[claim], n_results=top_k)
        return [{
            'statement': doc,
            'distance': results['distances'][0][i],
            'metadata': results['metadatas'][0][i]
        } for i, doc in enumerate(results['documents'][0])]

    # verify claim against retrieved facts
    def verify_claim(self, claim: str, facts: List[Dict]) -> Dict:
        if not facts:
            return {"verdict": "Unverifiable", "confidence": 0.0, "reasoning": "No relevant facts in database.", "evidence": []}

        similarities = [1 / (1 + f['distance']) for f in facts]
        best_sim = max(similarities)
        evidence = [f['statement'] for f in facts[:3]]

        if best_sim > 0.80:
            return {"verdict": "True", "confidence": round(best_sim, 3), "reasoning": "Strong match with official PIB records.", "evidence": evidence}
        elif best_sim > 0.55:
            return {"verdict": "Unverifiable", "confidence": round(best_sim, 3), "reasoning": "Related but not specific enough.", "evidence": evidence}
        else:
            return {"verdict": "False", "confidence": round(1 - best_sim, 3), "reasoning": "No match or contradicts official records.", "evidence": evidence}


    # main fact checking function
    def check_fact(self, user_input: str) -> Tuple[str, str]:
        claims = self.extract_claims(user_input)
        if not claims:
            return "⚠️ No verifiable claims found.", json.dumps({"error": "No claims"}, indent=2)

        summary_lines = [f"### Fact Check Results\n> {user_input[:200]}{'...' if len(user_input)>200 else ''}\n"]
        full_results = []

        for i, claim in enumerate(claims):
            is_vague, score = self.is_vague_claim(claim)

            if is_vague:
                summary_lines.append(f"{i+1}. 🤷‍♂️ **Unverifiable** – Too vague (add dates/names/numbers)")
                full_results.append({
                    "claim": claim, "verdict": "Unverifiable", "confidence": 0.0,
                    "reasoning": f"Too vague (score: {score:.2f}). Needs specifics.", "evidence": []
                })
                continue

            facts = self.retrieve_similar_facts(claim)
            result = self.verify_claim(claim, facts)
            result["claim"] = claim

            icon = "✅" if result["verdict"] == "True" else "❌" if result["verdict"] == "False" else "🤷‍♂️"
            short = claim[:90] + "..." if len(claim) > 90 else claim
            summary_lines.append(f"{i+1}. {icon} **{result['verdict']}** – {short}")
            full_results.append(result)

        return "\n".join(summary_lines), json.dumps({"original_text": user_input, "results": full_results}, indent=2, ensure_ascii=False)


# Initialising the fact check with csv that contains PIB data.
fact_checker = PIBFactChecker(default_csv="pib_facts.csv")


# User interface with Gradio.
def check_claim(text):
    if not text or not text.strip():
        return "Please enter a claim to verify.", ""
    return fact_checker.check_fact(text.strip())

with gr.Blocks(title="PIB Fact Checker", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 PIB Fact Checker
    Powered by official PIB data • 100% Offline • Instant Results
    """)

    text_input = gr.Textbox(
        placeholder="Paste any WhatsApp forward, news headline, or viral claim here...",
        lines=6,
        label="Enter the Claim to be verified"
    )
    check_btn = gr.Button("Check the fact Now", variant="primary", size="lg")

    verdict = gr.Markdown(label="Instant Verdict")
    details = gr.JSON(label="Detailed Report with Evidence")

    check_btn.click(check_claim, text_input, [verdict, details])

    gr.Markdown("### Verdicts: ✅ True • ❌ False • 🤷‍♂️ Unverifiable (includes vague claims)")

# Launch the Gradio app.
if __name__ == "__main__":
    demo.launch()