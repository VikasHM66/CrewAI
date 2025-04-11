from crewai import Task
from agents.data_loader_agent import DataLoaderAgent
from agents.data_cleaner_agent import DataCleanerAgent
from agents.exploratory_analysis_agent import ExploratoryAnalysisAgent
from agents.visualization_agent import VisualizationAgent
from agents.reporting_agent import ReportingAgent

class AnalysisTasks:
    @staticmethod
    def load_data_task(agent, file_path):
        return Task(
            description=f"Load and verify the data from {file_path}",
            agent=agent,
            expected_output="A confirmation that the data was loaded successfully with basic information about the dataset (number of rows, columns, etc.)",
        )

    @staticmethod
    def clean_data_task(agent, file_path):
        return Task(
            description=f"Clean and preprocess the data from {file_path}",
            agent=agent,
            expected_output="A cleaned version of the dataset with missing values handled and data types optimized",
        )

    @staticmethod
    def exploratory_analysis_task(agent, file_path):
        return Task(
            description=f"Perform exploratory data analysis on {file_path}",
            agent=agent,
            expected_output="Key statistics, correlations, and initial insights about the dataset",
        )

    @staticmethod
    def visualization_task(agent, file_path, plot_type='histogram'):
        return Task(
            description=f"Create {plot_type} visualizations for the data in {file_path}",
            agent=agent,
            expected_output="Path to saved visualization image file",
        )

    @staticmethod
    def answer_question_task(agent, file_path, question):
        return Task(
            description=f"Answer the question '{question}' about the data in {file_path}",
            agent=agent,
            expected_output="A clear and accurate answer to the question based on the data",
        )

    @staticmethod
    def generate_report_task(agent, file_path, question=None):
        return Task(
            description=f"Generate a comprehensive report about the data in {file_path}" + 
                       (f", including an answer to '{question}'" if question else ""),
            agent=agent,
            expected_output="A well-structured report with key findings, statistics, and visualizations",
        )