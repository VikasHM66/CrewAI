from crewai import Agent
from langchain.tools import tool
from tools.data_tools import DataTools
import numpy as np
from langchain_community.chat_models import ChatOpenAI  # Updated import path
import os
from langchain.llms import OpenAI  # Using the base OpenAI LLM instead
from langchain_community.chat_models import ChatOpenAI  # Correct import path

class ReportingAgent:
    @staticmethod
    def create():
        @tool
        def generate_report(file_path: str, question: str = None) -> str:
            """Generate comprehensive analysis report"""
            df = DataTools.load_data(file_path)
            analysis = DataTools.analyze_data(df)
            
            report = [
                f"=== Data Analysis Report ===",
                f"Dataset: {file_path}",
                f"Shape: {df.shape} (rows × columns)",
                "\n=== Column Overview ==="
            ]
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique = df[col].nunique()
                report.append(f"- {col}: {dtype} ({unique} unique values)")
            
            report.append("\n=== Key Statistics ===")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                stats = df[col].describe()
                report.append(
                    f"- {col}: "
                    f"mean={stats['mean']:.2f}, "
                    f"min={stats['min']:.2f}, "
                    f"max={stats['max']:.2f}"
                )
            
            if question:
                answer = DataTools.answer_question(df, question)
                report.append(f"\n=== Q&A ===\nQ: {question}\nA: {answer}")
            
            return "\n".join(report)

        return Agent(
            role="Senior Data Reporter",
            goal="Create clear analysis reports",
            backstory=(
                "You synthesize complex findings into "
                "actionable business insights."
            ),
            tools=[generate_report],
            verbose=True,
            llm=ChatOpenAI(model="deepseek-ai/deepseek-r1-distill-llama-8b",  # Your NVIDIA model
                           base_url="https://integrate.api.nvidia.com/v1",
                           api_key=os.getenv("NVIDIA_API_KEY"),  # Store in .env
                           temperature=0.6, top_p=0.7, max_tokens=4096),
            allow_delegation=True
        )