import streamlit as st
import fitz  
import os
import pandas as pd
import google.generativeai as genai
from langextract import Extractor


API_KEY = os.getenv("AIzaSyCxbkPfXQJECRyzMmS4gAFWPD-hab7G-EA")
genai.configure(api_key=API_KEY)



def extract_text_pymupdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def run_gemini_extraction(text, model="gemini-1.5-flash"):
    schema = {
        "title": "string",
        "authors": ["string"],
        "year": "string",
        "journal": "string",
        "objective": "string",
        "methodology": "string",
        "findings": "string",
        "limitations": "string",
    }

    extractor = Extractor(schema=schema, model=model, api_key=API_KEY)
    result = extractor.extract(text)
    return result



st.set_page_config(page_title="Literature Review AI", layout="wide")
st.title("📄 Literature Review AI")

with st.sidebar:
    st.header("⚙️ Settings")
    use_langextract = st.checkbox("Use LangExtract if available", value=True)
    model_choice = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    output_choice = st.radio("Export results to", ["CSV", "Google Sheets"], index=0)

uploaded_files = st.file_uploader("Upload up to 50 PDF research papers", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        st.write(f"Processing **{uploaded_file.name}** ...")

        # Extract text with PyMuPDF
        text = extract_text_pymupdf(uploaded_file)

        # Run extraction
        if use_langextract:
            extracted = run_gemini_extraction(text, model=model_choice)
        else:
            prompt = """
            Extract the following fields from the text:
            - Title
            - Authors
            - Year
            - Journal/Conference
            - Objective
            - Methodology
            - Findings
            - Limitations
            Return as JSON.
            """
            response = genai.GenerativeModel(model_choice).generate_content([text, prompt])
            extracted = response.candidates[0].content.parts[0].text

        if isinstance(extracted, str):
            try:
                import json
                extracted = json.loads(extracted)
            except:
                extracted = {"title": "", "authors": [], "year": "", "journal": "", "objective": "", "methodology": "", "findings": "", "limitations": ""}

        row = {
            "file_name": uploaded_file.name,
            **extracted,
            "snippet": text[:500]  # snippet for quick reference
        }

        results.append(row)

    # Display results
    df = pd.DataFrame(results)
    st.subheader("📊 Extracted Results")
    st.dataframe(df[["file_name", "title", "authors", "year", "journal"]])

    # Per-row details with source snippet
    st.subheader("🔍 Detailed View with Source Snippets")
    for r in results:
        with st.expander(r["file_name"]):
            st.write("**Title:**", r.get("title", ""))
            st.write("**Authors:**", ", ".join(r.get("authors", [])))
            st.write("**Year:**", r.get("year", ""))
            st.write("**Journal:**", r.get("journal", ""))
            st.write("**Objective:**", r.get("objective", ""))
            st.write("**Methodology:**", r.get("methodology", ""))
            st.write("**Findings:**", r.get("findings", ""))
            st.write("**Limitations:**", r.get("limitations", ""))
            st.code(r["snippet"], language="text")

    # Export
    if output_choice == "CSV":
        st.download_button("⬇️ Download Results as CSV", df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")
    elif output_choice == "Google Sheets":
        st.info("Google Sheets integration pending setup.")
