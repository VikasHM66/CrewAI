from crewai import Agent
from langchain.tools import tool
from tools.data_tools import DataTools
from langchain_community.chat_models import ChatOpenAI  # Updated import path
import os
from langchain.llms import OpenAI  # Using the base OpenAI LLM instead
from langchain_community.chat_models import ChatOpenAI  # Correct import path

class DataLoaderAgent:
    @staticmethod
    def create():
        @tool
        def load_data(file_path: str) -> str:
            """Load and validate data from a CSV file"""
            df = DataTools.load_data(file_path)
            return (
                f"Data loaded successfully. "
                f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
                f"Columns: {', '.join(df.columns)}"
            )

        return Agent(
            role="Senior Data Loader",
            goal="Load and validate data files",
            backstory=(
                "You are an expert in data loading and initial validation. "
                "You ensure data is properly loaded before analysis."
            ),
            tools=[load_data],
            verbose=True,
            llm=ChatOpenAI(model="deepseek-ai/deepseek-r1-distill-llama-8b",  # Your NVIDIA model
                           base_url="https://integrate.api.nvidia.com/v1",
                           api_key=os.getenv("NVIDIA_API_KEY"),  # Store in .env
                           temperature=0.6, top_p=0.7, max_tokens=4096),            
            allow_delegation=False
        )