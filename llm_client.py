# llm_client.py

import os
import logging
from typing import Optional, Dict
from pydantic import BaseModel, Field

from langchain import LLMChain, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import GoogleGenerativeAI

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Metadata Model
# ----------------------------
class PaperMetadata(BaseModel):
    title: Optional[str] = Field(None, description="Paper title")
    authors: Optional[str] = Field(None, description="Comma-separated authors or list")
    publication_year: Optional[str] = Field(None, description="Year of publication")
    journal_or_conference: Optional[str] = Field(None, description="Name of journal or conference")
    research_objective: Optional[str] = Field(None, description="Research objective")
    methodology: Optional[str] = Field(None, description="Methodology summary")
    key_findings: Optional[str] = Field(None, description="Key findings")
    limitations: Optional[str] = Field(None, description="Limitations")

# ----------------------------
# Prompt Template
# ----------------------------
EXTRACTION_PROMPT = """
You are an AI metadata extractor. Extract the following fields from the research paper text.
Return the output as JSON matching the schema exactly.

Paper text:
{text}

Fields to extract:
- title
- authors
- publication_year
- journal_or_conference
- research_objective
- methodology
- key_findings
- limitations
"""

# ----------------------------
# Gemini Extractor
# ----------------------------
class GeminiExtractor:
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY environment variable not set.")
        
        # Initialize Google Generative AI LLM
        self.llm = GoogleGenerativeAI(model=model, temperature=temperature)
        
        # Prompt template and LangChain parser
        self.prompt = PromptTemplate(template=EXTRACTION_PROMPT, input_variables=["text"])
        self.parser = PydanticOutputParser(pydantic_object=PaperMetadata)
        self.prompt_with_format = PromptTemplate(
            template=EXTRACTION_PROMPT + "\n\nJSON Schema format instructions:\n{format_instructions}",
            input_variables=["text", "format_instructions"],
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_with_format)

    def extract(self, text: str) -> Dict:
        """
        Extract metadata from a text string using Gemini + LangChain.
        Returns a dictionary matching PaperMetadata fields.
        """
        format_instructions = self.parser.get_format_instructions()
        try:
            response = self.chain.run({"text": text, "format_instructions": format_instructions})
            parsed = self.parser.parse(response)
            return parsed.dict()
        except Exception as e:
            logger.exception("LLM extraction failed")
            # Return empty/default metadata with error info
            return {
                "title": None,
                "authors": None,
                "publication_year": None,
                "journal_or_conference": None,
                "research_objective": None,
                "methodology": None,
                "key_findings": None,
                "limitations": None,
                "error": f"LLM extraction failed: {e}"
            }
