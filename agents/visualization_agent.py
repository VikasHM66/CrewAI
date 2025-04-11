from crewai import Agent
from langchain.tools import tool
from tools.data_tools import DataTools
import os
from langchain_community.chat_models import ChatOpenAI  # Updated import path
import os
from langchain.llms import OpenAI  # Using the base OpenAI LLM instead
from langchain_community.chat_models import ChatOpenAI  # Correct import path

class VisualizationAgent:
    @staticmethod
    def create():
        @tool
        def create_visualization(file_path: str, plot_type: str = 'histogram') -> str:
            """Generate data visualizations"""
            df = DataTools.load_data(file_path)
            image_path = DataTools.visualize_data(df, plot_type)
            return (
                f"Visualization created: {plot_type}\n"
                f"Saved to: {os.path.abspath(image_path)}"
            )

        return Agent(
            role="Data Visualization Expert",
            goal="Create insightful visualizations",
            backstory=(
                "You transform complex data into clear, "
                "informative visual representations."
            ),
            tools=[create_visualization],
            verbose=True,
            llm=ChatOpenAI(model="deepseek-ai/deepseek-r1-distill-llama-8b",  # Your NVIDIA model
                           base_url="https://integrate.api.nvidia.com/v1",
                           api_key=os.getenv("NVIDIA_API_KEY"),  # Store in .env
                           temperature=0.6, top_p=0.7, max_tokens=4096),
            allow_delegation=False
        )