from crewai import Agent
from langchain.tools import tool
from tools.data_tools import DataTools
from langchain_community.chat_models import ChatOpenAI  # Updated import path
import os
from langchain.llms import OpenAI  # Using the base OpenAI LLM instead
from langchain_community.chat_models import ChatOpenAI  # Correct import path

class ExploratoryAnalysisAgent:
    @staticmethod
    def create():
        @tool
        def analyze_data(file_path: str) -> str:
            """Perform exploratory data analysis"""
            df = DataTools.load_data(file_path)
            analysis = DataTools.analyze_data(df)
            return (
                f"Analysis complete. Key findings:\n"
                f"- Numeric columns: {[col for col in df.select_dtypes(include=['number']).columns]}\n"
                f"- Categorical columns: {[col for col in df.select_dtypes(include=['category', 'object']).columns]}\n"
                f"- Correlation matrix available"
            )

        @tool
        def answer_question(file_path: str, question: str) -> str:
            """Answer specific questions about the data"""
            df = DataTools.load_data(file_path)
            return DataTools.answer_question(df, question)

        return Agent(
            role="Senior Data Analyst",
            goal="Discover insights through EDA",
            backstory=(
                "You are an experienced analyst who finds patterns "
                "and relationships in complex datasets."
            ),
            tools=[analyze_data, answer_question],
            verbose=True,
            llm=ChatOpenAI(model="deepseek-ai/deepseek-r1-distill-llama-8b",  # Your NVIDIA model
                           base_url="https://integrate.api.nvidia.com/v1",
                           api_key=os.getenv("NVIDIA_API_KEY"),  # Store in .env
                           temperature=0.6, top_p=0.7, max_tokens=4096),
            allow_delegation=False
        )