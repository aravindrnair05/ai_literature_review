
import streamlit as st
import pandas as pd
import tempfile
import os
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_client import GeminiExtractor


# ----------------------------
# Streamlit App
# ----------------------------

import streamlit as st

st.set_page_config(page_title="🧠📄🔍 Research Article Data Extractor")
st.title("🧠📄🔍 Research Article Data Extractor")


# ----------------------------
# Helper function
# ----------------------------
def process_single_file(filename, file_bytes, extractor):
    """
    Save uploaded PDF temporarily, extract text, and run GeminiExtractor.
    """
    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    text = ""
    try:
        reader = PdfReader(temp_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        text = f"Error extracting text from {filename}: {e}"

    # Extract metadata via Gemini
    try:
        metadata = extractor.extract(text)
    except Exception as e:
        metadata = {
            "title": None,
            "authors": None,
            "year": None,
            "journal": None,
            "objective": None,
            "methodology": None,
            "findings": None,
            "limitations": None,
            "error": str(e),
        }

    # Clean up
    os.remove(temp_path)

    return {"filename": filename, **metadata}


st.write(
    "Upload up to 50 PDF research papers."
)

# File upload widget
uploaded_files = st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

# Validate file count
if uploaded_files and len(uploaded_files) > 50:
    st.error("⚠️ You can upload a maximum of 50 files.")
    uploaded_files = uploaded_files[:50]

if uploaded_files:
    st.write(f"**{len(uploaded_files)} file(s) uploaded**:")
    for f in uploaded_files:
        st.write("- " + f.name)

    if st.button("🔍 Process Files"):
        extractor = GeminiExtractor()
        results = []

        progress = st.progress(0)
        total = len(uploaded_files)

        # Process in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_file, f.name, f.read(), extractor): f.name
                for f in uploaded_files
            }

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                progress.progress((i + 1) / total)

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        st.success("✅ Metadata extraction complete!")

        # Show table
        st.dataframe(df, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            csv,
            "publication_metadata.csv",
            "text/csv",
            key="download-csv",
        )
else:
    st.info("Please upload one or more PDF files to begin.")
