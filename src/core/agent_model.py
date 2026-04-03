"""Agentic AI logic using Fusion-RAG (Clinical + Economic + Mortality)."""

import pandas as pd
import logging
from pathlib import Path
from langchain_ollama import ChatOllama
from src.core.rag_engine import SubstanceFusionRAG
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SubstanceRiskAgent:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.rag = SubstanceFusionRAG()
        try:
            self.llm = ChatOllama(model=model_name, temperature=0.1)
            logger.info(f"Initialized Local LLM: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama: {e}")
            self.llm = None

    def run_query(self, user_query: str):
        logger.info(f"Reasoning over Fusion-RAG for: {user_query}")
        
        # 1. Clinical context
        try:
            clinical_res = self.rag.query(user_query, collection="clinical_reviews", n_results=5)
            clinical_context = ""
            if clinical_res['documents'][0]:
                for i, doc in enumerate(clinical_res['documents'][0]):
                    sentiment = clinical_res['metadatas'][0][i].get('sentiment', 'Unknown')
                    clinical_context += f"- Patient Voice ({sentiment}): {doc[:300]}...\n"
            else:
                clinical_context = "No specific clinical reviews found."
        except Exception:
            clinical_context = "Clinical layer unavailable."

        # 2. Economic/Mortality context
        try:
            intel_res = self.rag.query(user_query, collection="substance_intelligence", n_results=3)
            intel_context = ""
            if intel_res['documents'][0]:
                for doc in intel_res['documents'][0]:
                    intel_context += f"- Intel Signal: {doc}\n"
            else:
                intel_context = "No specific economic or mortality signals found locally."
        except Exception:
            intel_context = "Intelligence layer unavailable."

        # 3. LLM Synthesis
        prompt = f"""
        [CONTEXT: NEXUS-CORTEX FUSION INTELLIGENCE]
        USER QUERY: {user_query}
        
        [LAYER 1: CLINICAL PATIENT EXPERIENCE]
        {clinical_context}
        
        [LAYER 2: ECONOMIC & MORTALITY SIGNALS]
        {intel_context}
        
        [INSTRUCTION]
        You are a clinical intelligence agent. Provide a 'Triangulated Risk Profile' for the query.
        Synthesize patient-reported sentiment with the hard economic and mortality signals provided.
        Professional, evidence-based, and concise. Do not make up facts.
        """
        
        if not self.llm:
            return f"**LLM Connection Error:** Start 'ollama serve'. Context retrieved: {intel_context}"

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"**Inference Error:** {e}\n\n**CONTEXT:**\n{intel_context}\n\n{clinical_context}"
