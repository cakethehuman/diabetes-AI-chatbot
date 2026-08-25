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

> 💡 **Note:** If your primary execution script uses a different name (e.g., `main.py` or `chatbot.py`), replace `app.py` with your actual file name.

Upon a successful launch, Streamlit will automatically open your default web browser to the following address:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.5:8501
```

---
