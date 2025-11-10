# Elysian Analytics: AI-Powered eDNA Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://elysian-analytics-7lr7c8jvkadnuykxlteokg.streamlit.app/)

[Demo Video](https://www.youtube.com/watch?v=KUIW1Y8ABew)

Elysian Analytics is a powerful Streamlit web application built to accelerate the analysis of environmental DNA (eDNA) sequences. It provides an AI-driven pipeline to rapidly classify eukaryotic taxa and identify potentially novel organisms, overcoming the limitations of traditional, time-consuming, database-dependent methods.

This tool integrates machine learning models with live NCBI BLAST searches to provide a comprehensive, interactive, and actionable biodiversity report from a simple FASTA file.

---

## Problem & Solution

Analyzing raw eDNA sequencing reads, especially from under-sampled environments like the deep sea, presents two major challenges:

1.  **Poor Database Representation:** Reference databases (like NCBI, SILVA, PR2) are often incomplete, leading to many unassigned or misclassified sequences.
2.  **Computational Bottlenecks:** Traditional methods like BLAST are computationally expensive and slow, hindering rapid analysis.

**Elysian Analytics solves this** by using a hybrid approach:

* **AI-First Classification:** A trio of lightweight AI models (CNN, XGBoost, Random Forest) provide near-instant taxonomic predictions based on sequence patterns.
* **Integrated Novelty Check:** The app cross-references the AI's predictions with a live NCBI BLAST search, automatically flagging sequences that are:
    * **AI Discoveries:** Novel patterns recognized by the AI that have no match in NCBI.
    * **Potentially Novel:** Sequences with low database matches.
    * **Consistent:** Sequences where the AI and NCBI agree.

## Key Features 

* **Multi-Model Analysis:** Choose between three different trained AI models:
    * **Deep Learning (1D-CNN):** A Keras/TensorFlow model for deep feature extraction.
    * **XGBoost:** A fast and highly accurate gradient-boosting model (Recommended).
    * **Random Forest:** A robust baseline for k-mer based classification.
* **Live NCBI BLAST Integration:** Automatically runs a `blastn` query on your sequences to provide real-time novelty and percent identity metrics.
* **Integrated Insights:** Generates a unified report with a "Remarks" column that instantly tells you if a sequence is a novel discovery, a potential new species, or a known organism.
* **Interactive Dashboard:** The results page features key metrics (Total ASVs, AI Discoveries) and interactive Plotly pie charts to visualize taxonomic distribution.
* **Exportable Results:** Download the complete, integrated analysis report as a `.csv` file with one click.
* **User-Friendly Interface:** A clean, multi-page Streamlit interface with a persistent sidebar for easy navigation and file upload.

---

## Technology Stack 

* **Frontend:** Streamlit
* **Core Python:** Python 3.9+
* **Data Processing:** Pandas, NumPy
* **Bioinformatics:** BioPython (for FASTA parsing & NCBI BLAST)
* **Machine Learning:** Scikit-learn (Random Forest, LabelEncoder), XGBoost
* **Deep Learning:** TensorFlow / Keras
* **Visualization:** Plotly Express

---

## Project Structure 

For the app to run correctly, you must have the trained models and static files in the following structure:
```text
eDNA_Analytics/
├── app.py              
├── models/          
│   ├── xgboost_model.pkl
│   ├── random_forest_baseline.pkl
│   └── label_encoder.pkl
├── dl_model.h5        
├── dna_helix.png    
├── style.css         
└── requirements.txt
```
##  Getting Started

### 1. Prerequisites

* Python 3.9 or higher
* A virtual environment (recommended)

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Elysian-Analytics.git](https://github.com/your-username/Elysian-Analytics.git)
    cd Elysian-Analytics
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the contents below, then run `pip install -r requirements.txt`.

    **`requirements.txt`**
    ```
    streamlit
    pandas
    numpy
    tensorflow
    scikit-learn
    xgboost
    plotly
    biopython
    ```

4.  **Add Model & Static Files:**
    Place your trained models (`.pkl`, `.h5`), background image (`dna_helix.png`), and CSS (`style.css`) in the correct directories as shown in the **Project Structure** section.

### 3. Running the App

Once all dependencies and model files are in place, run the app from your terminal:

```bash
streamlit run app.py
