# llm_client.py
from typing import Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperMetadata(BaseModel):
    title: Optional[str] = Field(None, description="Paper title")
    authors: Optional[str] = Field(None, description="Comma-separated authors or list")
    publication_year: Optional[str] = Field(None, description="Year of publication")
    journal_or_conference: Optional[str] = Field(None, description="Name of journal or conference")
    research_objective: Optional[str] = Field(None, description="Research objective")
    methodology: Optional[str] = Field(None, description="Methodology summary")
    key_findings: Optional[str] = Field(None, description="Key findings")
    limitations: Optional[str] = Field(None, description="Limitations")


EXTRACTION_PROMPT = """
You are a metadata extractor. Given the text of a research publication, extract the following fields:

- Title
- Authors
- Year of publication
- Journal or Conference
- Research Objective
- Methodology
- Key Findings
- Limitations

Return the result strictly in JSON format matching the schema below.

Text:
{text}
"""


class GeminiExtractor:
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY environment variable not set")

        # ✅ Explicitly pass API key to avoid metadata service lookup
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
        )

        self.parser = PydanticOutputParser(pydantic_object=PaperMetadata)

        self.prompt = PromptTemplate(
            template=EXTRACTION_PROMPT + "\n\nOutput JSON schema: {format_instructions}",
            input_variables=["text", "format_instructions"],
        )

        # ✅ Replace LLMChain with LangChain’s new “Runnable” API
        self.chain = self.prompt | self.llm | self.parser

    def extract(self, text: str) -> Dict:
        format_instructions = self.parser.get_format_instructions()
        try:
            response = self.chain.invoke({"text": text, "format_instructions": format_instructions})
            return response.dict()
        except Exception as e:
            logger.exception("LLM extraction failed")
            return {"error": f"LLM extraction failed: {e}"}
