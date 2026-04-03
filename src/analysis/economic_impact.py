"""
Economic Impact Analysis Module for Nexus-Cortex.
Models the societal burden of the 'Big Three': Alcohol, Tobacco, and Illicit Drugs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_economic_data():
    """
    Generates high-fidelity economic impact data based on 2024 National Reports.
    Sources: White House Economic Report (CEA), CDC, and NIDA estimates.
    """
    logger.info("Generating economic impact intelligence data...")
    
    # 1. Pillar Summary (The 'Big Three')
    pillars = [
        {"type": "Pillar Summary", "subcategory": "Illicit Drugs", "value": 740.0, "impact": "Opioids, Cocaine, Methamphetamine Total Societal Cost."},
        {"type": "Pillar Summary", "subcategory": "Tobacco", "value": 606.0, "impact": "Cigarette Smoking (Smoking-related illness + Productivity)."},
        {"type": "Pillar Summary", "subcategory": "Alcohol", "value": 249.0, "impact": "Binge Drinking, DUI, and Chronic Health Costs."},
    ]
    
    # 2. Detailed Cost Allocation (Illicit Drugs)
    drugs_detail = [
        {"type": "Illicit Drugs", "category": "Direct Costs", "subcategory": "Healthcare", "value": 107.0, "impact": "ER visits, OD treatment, Hospitalizations."},
        {"type": "Illicit Drugs", "category": "Direct Costs", "subcategory": "Criminal Justice", "value": 45.0, "impact": "Policing, Incarceration, Court costs."},
        {"type": "Illicit Drugs", "category": "Indirect Costs", "subcategory": "Productivity Loss", "value": 92.6, "impact": "Absenteeism, Disability, Incarceration-related drain."},
        {"type": "Illicit Drugs", "category": "Indirect Costs", "subcategory": "Loss of Life (VSL)", "value": 1110.0, "impact": "Value of Statistical Life lost to fatal overdoses."},
        {"type": "Illicit Drugs", "category": "Productivity Loss Detail", "subcategory": "Labor Participation", "value": 65.0, "impact": "Direct drain on active workforce."},
        {"type": "Illicit Drugs", "category": "Productivity Loss Detail", "subcategory": "Incarceration", "value": 12.0, "impact": "Economic drain from drug-related prison time."},
        {"type": "Illicit Drugs", "category": "Quality of Life", "subcategory": "Quality of Life Loss", "value": 485.4, "impact": "Non-fatal morbidity impact on societal well-being."}
    ]
    
    # 3. Detailed Cost Allocation (Tobacco / Smoking vs Vaping)
    tobacco_detail = [
        {"type": "Tobacco", "category": "Direct Costs", "subcategory": "Healthcare", "value": 241.0, "impact": "Smoking-related cancer, heart disease, COPD treatment."},
        {"type": "Tobacco", "category": "Indirect Costs", "subcategory": "Productivity Loss", "value": 365.0, "impact": "Premature death and illness-related work loss."},
        {"type": "Tobacco", "category": "Productivity Loss Detail", "subcategory": "Smoking (Cigarettes)", "value": 354.1, "impact": "Legacy health drain from traditional smoking."},
        {"type": "Tobacco", "category": "Productivity Loss Detail", "subcategory": "Vaping/E-cigarettes", "value": 10.9, "impact": "Rising costs associated with youth vaping and lung injury."}
    ]
    
    # 4. Detailed Cost Allocation (Alcohol)
    alcohol_detail = [
        {"type": "Alcohol", "category": "Direct Costs", "subcategory": "Healthcare", "value": 28.0, "impact": "Liver disease, Alcohol poisoning, injury treatment."},
        {"type": "Alcohol", "category": "Direct Costs", "subcategory": "Crash Costs (DUI)", "value": 13.0, "impact": "Property damage and medical costs from alcohol-involved crashes."},
        {"type": "Alcohol", "category": "Indirect Costs", "subcategory": "Productivity Loss", "value": 179.0, "impact": "Hangovers, workplace injury, and premature mortality."},
    ]

    # 5. Per-User Analysis (Micro-Economic)
    per_user = [
        {"type": "Per-User Analysis", "category": "Healthcare", "subcategory": "Cigarette Smoker", "value": 8000, "impact": "Annual excess medical expenditure per smoker."},
        {"type": "Per-User Analysis", "category": "Healthcare", "subcategory": "Exclusive Vaper", "value": 1800, "impact": "Annual excess medical expenditure per vaper."},
        {"type": "Per-User Analysis", "category": "Healthcare", "subcategory": "Dual User", "value": 2050, "impact": "Increased risk profile for poly-tobacco users."}
    ]
    
    all_records = pillars + drugs_detail + tobacco_detail + alcohol_detail + per_user
    df = pd.DataFrame(all_records)
    
    save_path = DATA_DIR / "processed_economic_costs.parquet"
    df.to_parquet(save_path)
    logger.info(f"Economic intelligence dataset saved to {save_path}")

if __name__ == "__main__":
    generate_economic_data()
