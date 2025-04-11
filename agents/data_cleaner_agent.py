from crewai import Agent
from langchain.tools import tool
from tools.data_tools import DataTools
from langchain_community.chat_models import ChatOpenAI  # Updated import path
import os
from langchain.llms import OpenAI  # Using the base OpenAI LLM instead
from langchain_community.chat_models import ChatOpenAI  # Correct import path

class DataCleanerAgent:
    @staticmethod
    def create():
        @tool
        def clean_data(file_path: str) -> str:
            """Clean and preprocess raw data"""
            df = DataTools.load_data(file_path)
            cleaned_df = DataTools.clean_data(df)
            return (
                f"Data cleaned successfully. "
                f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}\n"
                f"Missing values handled, data types optimized."
            )

        return Agent(
            role="Data Cleaning Specialist",
            goal="Prepare raw data for analysis",
            backstory=(
                "You are meticulous about data quality. "
                "You handle missing values, outliers, and formatting issues."
            ),
            tools=[clean_data],
            verbose=True,
            llm=ChatOpenAI(model="deepseek-ai/deepseek-r1-distill-llama-8b",  # Your NVIDIA model
                           base_url="https://integrate.api.nvidia.com/v1",
                           api_key=os.getenv("NVIDIA_API_KEY"),  # Store in .env
                           temperature=0.6,top_p=0.7, max_tokens=4096),
            allow_delegation=False
        )