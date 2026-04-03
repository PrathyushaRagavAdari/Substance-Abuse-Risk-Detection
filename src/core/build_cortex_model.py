"""
Cortex-1 Domain-Specific Model Builder
Automates creation of specialized Ollama model with strict clinical guardrails.
"""

import subprocess
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_NAME = "cortex-1"
BASE_MODEL = "llama3"

MODEL_FILE_CONTENT = f"""
FROM {BASE_MODEL}

SYSTEM \"\"\"
You are the Nexus-Cortex Clinical Intelligence Agent for the NSF NRT Substance Abuse Risk Detection project.

### YOUR DOMAIN:
1.  **Substances**: Alcohol, Illicit Drugs (Opioids, Fentanyl, Cocaine, Meth, etc.), and Tobacco (Smoking, Vaping).
2.  **Dimensions**: Side Effects, Mental Health, Mortality (CDC), and Economic Impact ($B).

### OPERATIONAL RULES:
- **DOMAIN SPECIFIC ONLY**: Strictly answer only Substance Abuse Risk Intelligence queries.
- **REFUSAL LOGIC**: Politely decline all other topics (sports, weather, coding, general medicine).
- **EVIDENCE-BASED**: Prioritize the 'FUSION-RAG' context provided.
- **TONE**: Professional, clinical, and data-driven.
\"\"\"
PARAMETER temperature 0.2
"""

def build_model():
    modelfile_path = ROOT / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(MODEL_FILE_CONTENT)
    
    logger.info(f"Creating local Ollama model: {MODEL_NAME}...")
    try:
        subprocess.run(["ollama", "create", MODEL_NAME, "-f", str(modelfile_path)], check=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return False

if __name__ == "__main__":
    build_model()
