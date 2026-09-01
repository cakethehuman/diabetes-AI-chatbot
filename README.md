<h1 align="center"> Diabetes AI Chat-bot & Prediction</h1>

<p align="center">
  <b>An interactive web application built with Streamlit to predict diabetes risk factors and provide helpful AI-guided health insights.</b>
</p>

---

## 📌 Table of Contents
1. [📋 Prerequisites](#-prerequisites)
2. [🛠️ Installation & Setup](#️-installation--setup)
3. [🚀 Running the Application](#-running-the-application)

---

## 📋 Prerequisites
Before running the application, ensure your local machine has the following installed:
* [Python 3.11](https://python.org) or higher
* `pip` (Python package manager).
---

## 🛠️ Installation & Setup

Follow these steps to configure your local development environment:

1. **Clone the Repository**
   ```bash
   git clone [github link] (Copy from the page)
   cd diabetes-AI-chatbot
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Libraries**
   Install Streamlit, machine learning modules, and chatbot dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

Once the installation is complete, launch the local Streamlit server using the following command:

```bash
streamlit run app.py (Run streamlit app and wait till .chromadb folder appear)
python -m src.scripts.buildIndex (To insert documents to chromadb)
```
Or you could use docker instaed

```bash
docker compose up -d --build
```

> 💡 **Note:** If your primary execution script uses a different name (e.g., `main.py` or `chatbot.py`), replace `app.py` with your actual file name.

Upon a successful launch, Streamlit will automatically open your default web browser to the following address:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.5:8501
```

---

## 🧠 System Architecture & Data Flow

This application links a traditional Machine Learning classification model with an intelligent Generative AI system using **Retrieval-Augmented Generation (RAG)**.
<img width="1874" height="812" alt="Untitled Diagram drawio (1)" src="https://github.com/user-attachments/assets/0b068109-5b70-4d5e-b657-4f808352c318" />


### Process Lifecycle:
1. **User Input:** The patient enters clinical parameters via the Streamlit interface.
2. **ML Prediction:** The classification model processes the parameters.
3. **Data Extraction:** The engine outputs two crucial variables:
   * **Prediction Result:** Diagnostic classification (`Have diabetes` / `Dont Have diabetes`).
   * **Probability Score (`proba`):** The exact statistical certainty percentage of the risk calculation.
4. **Context Injection & Retrieval:** The prediction result and probability metrics are automatically injected into the AI Agent's active memory buffer while the system dynamically pulls relevant context from local medical documents/
5. **AI Synthesis:** The agent aggregates the user's specific risk numbers with guidelines from verified medical literature to deliver precise, tailored conversational responses.

---

