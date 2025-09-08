# AI Publication Analyzer

A Streamlit web application that extracts structured metadata from research PDFs using Google Gemini (Generative AI) via LangChain.

## Features
- Upload up to 50 PDF files.
- Extract title, authors, publication year, journal/conference, research objective, methodology, key findings, and limitations.
- Structured extraction using LangChain & Pydantic output parsing.
- Download final results as CSV.
- Caching to prevent duplicate processing.
- Error handling for invalid or scanned PDFs and API issues.

## Requirements
- Python 3.9+
- Google API key with access to Gemini (create one at Google AI Studio).
- (Optional) Vertex AI credentials if running on Vertex.

### Install dependencies
```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
