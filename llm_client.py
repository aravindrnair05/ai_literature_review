# llm_client.py
from typing import Optional, Dict
from pydantic import BaseModel, Field
from langchain import LLMChain, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import GoogleGenerativeAI
import logging

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
You are a metadata extractor. Given the text of a research publication, extract the following fields in JSON...
{text}
"""

class GeminiExtractor:
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        self.llm = GoogleGenerativeAI(model=model, temperature=temperature)
        self.prompt = PromptTemplate(template=EXTRACTION_PROMPT, input_variables=["text"])
        self.parser = PydanticOutputParser(pydantic_object=PaperMetadata)
        self.prompt_with_format = PromptTemplate(
            template=EXTRACTION_PROMPT + "\n\nOutput JSON schema: {format_instructions}",
            input_variables=["text", "format_instructions"],
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_with_format)

    def extract(self, text: str) -> Dict:
        format_instructions = self.parser.get_format_instructions()
        try:
            response = self.chain.run({"text": text, "format_instructions": format_instructions})
            parsed = self.parser.parse(response)
            return parsed.dict()
        except Exception as e:
            logger.exception("LLM extraction failed")
            return {"error": f"LLM extraction failed: {e}"}
