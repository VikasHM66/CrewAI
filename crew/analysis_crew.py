from crewai import Crew
from agents.data_loader_agent import DataLoaderAgent
from agents.data_cleaner_agent import DataCleanerAgent
from agents.exploratory_analysis_agent import ExploratoryAnalysisAgent
from agents.visualization_agent import VisualizationAgent
from agents.reporting_agent import ReportingAgent
from tasks.analysis_tasks import AnalysisTasks

class AnalysisCrew:
    def __init__(self):
        self.data_loader = DataLoaderAgent.create()
        self.data_cleaner = DataCleanerAgent.create()
        self.analyst = ExploratoryAnalysisAgent.create()
        self.visualizer = VisualizationAgent.create()
        self.reporter = ReportingAgent.create()

    def analyze_data(self, file_path, question=None):
        # Create tasks
        load_task = AnalysisTasks.load_data_task(self.data_loader, file_path)
        clean_task = AnalysisTasks.clean_data_task(self.data_cleaner, file_path)
        analysis_task = AnalysisTasks.exploratory_analysis_task(self.analyst, file_path)
        report_task = AnalysisTasks.generate_report_task(self.reporter, file_path, question)
        
        # Visualization task - we'll let the reporter decide if needed
        visualization_task = AnalysisTasks.visualization_task(
            self.visualizer, file_path, 'histogram'
        )
        
        # Form the crew
        crew = Crew(
            agents=[
                self.data_loader,
                self.data_cleaner,
                self.analyst,
                self.visualizer,
                self.reporter
            ],
            tasks=[
                load_task,
                clean_task,
                analysis_task,
                visualization_task,
                report_task
            ],
            verbose=2
        )
        
        # Execute the crew's work
        result = crew.kickoff()
        return result