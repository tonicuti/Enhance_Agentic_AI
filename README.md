# Agentic RAG Prototype

## Run the Streamlit interface

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The current page provides the chat layout and backend selector. Reranker on/off comparisons belong to benchmark runs; the selected reranker configuration is fixed for the chat pipeline. Retrieval and answer generation are not connected yet; submitted questions show a setup notice rather than a generated answer.
