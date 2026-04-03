"""
Nexus-Cortex Fusion RAG Engine
Ingests Multi-Source Intelligence (Clinical + Economic + Mortality) into ChromaDB.
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "chroma_db"
# Source Paths
REVIEWS_PATH = ROOT / "data" / "processed" / "processed_drug_reviews.parquet"
ECON_PATH = ROOT / "data" / "processed" / "processed_economic_costs.parquet"
MORTALITY_PATH = ROOT / "data" / "processed" / "processed_drug_specific.parquet"

class SubstanceFusionRAG:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # Clinical Collection (Patient Reviews)
        self.clinical_col = self.client.get_or_create_collection(
            name="clinical_reviews",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Intelligence Collection (Economic + Mortality Meta-data)
        self.intel_col = self.client.get_or_create_collection(
            name="substance_intelligence",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def build_fusion_database(self, sample_size: int = 5000):
        """Orchestrate the multi-source ingestion process."""
        logger.info("🚀 Building Fusion-RAG Intelligence Layer...")
        self._load_clinical_reviews(sample_size)
        self._load_economic_intelligence()
        self._load_mortality_intelligence()
        logger.info("✨ Fusion-RAG Build Complete.")

    def _load_clinical_reviews(self, sample_size):
        """Ingest patient reviews (Clinical Layer)."""
        if not REVIEWS_PATH.exists():
            return
        df = pd.read_parquet(REVIEWS_PATH)
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        
        logger.info(f"Indexing {len(df)} clinical reviews...")
        ids = df["uniqueID"].astype(str).tolist()
        documents = df["review"].tolist()
        metadatas = df[["drugName", "condition", "rating", "sentiment"]].to_dict(orient="records")
        
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.clinical_col.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

    def _load_economic_intelligence(self):
        """Ingest economic impact metrics (Financial Layer)."""
        if not ECON_PATH.exists():
            return
        df = pd.read_parquet(ECON_PATH)
        # Filter for the comparative pillars and analytics
        df_metric = df[df['type'].isin(['Illicit Drugs', 'Tobacco', 'Alcohol', 'Per-User Analysis'])]
        
        logger.info(f"Indexing {len(df_metric)} economic impact signals...")
        ids = [f"econ_{sub}_{cat}_{i}" for i, (sub, cat) in enumerate(zip(df_metric['type'], df_metric['subcategory']))]
        
        # Build descriptive documents for semantic retrieval
        documents = [f"Substance: {r['type']}. Domain: {r['subcategory']}. Financial Impact: ${r['value']}B. Description: {r['impact']}" for _, r in df_metric.iterrows()]
        metadatas = [{"source": "Federal Economic Report", "type": r['type'], "subcategory": r['subcategory']} for _, r in df_metric.iterrows()]
        
        self.intel_col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def _load_mortality_intelligence(self):
        """Ingest mortality trends (CDC Layer)."""
        if not MORTALITY_PATH.exists():
            return
        df = pd.read_parquet(MORTALITY_PATH)
        df_us = df[df["jurisdiction"] == "United States"].sort_values("timestamp").groupby("indicator").last().reset_index()
        
        logger.info(f"Indexing {len(df_us)} national mortality hotspots...")
        ids = [f"mortality_{i}" for i in range(len(df_us))]
        documents = [f"Substance: {r['indicator']}. Annual deaths: {r['value']}. Status: {r['timestamp']}." for _, r in df_us.iterrows()]
        metadatas = [{"source": "CDC WONDER", "indicator": r['indicator']} for _, r in df_us.iterrows()]
        
        self.intel_col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query: str, collection: str = "clinical_reviews", n_results: int = 5):
        """Retrieve context from clinical or intelligence collections."""
        target = self.clinical_col if collection == "clinical_reviews" else self.intel_col
        return target.query(query_texts=[query], n_results=n_results)

if __name__ == "__main__":
    fusion = SubstanceFusionRAG()
    fusion.build_fusion_database()
