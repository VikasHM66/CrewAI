import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import numpy as np
import logging

class DataTools:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning
        """
        try:
            # Handle missing values
            df = df.dropna(how='all')
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Convert object types to categorical where appropriate
            for col in df.select_dtypes(include=['object']):
                if len(df[col].unique()) / len(df[col]) < 0.5:
                    df[col] = df[col].astype('category')
            
            logging.info("Data cleaning completed successfully")
            return df
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise

    @staticmethod
    def analyze_data(df: pd.DataFrame) -> dict:
        """
        Perform basic statistical analysis
        """
        analysis = {
            'summary': df.describe(include='all').to_dict(),
            'correlations': df.select_dtypes(include=[np.number]).corr().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        return analysis

    @staticmethod
    def visualize_data(df: pd.DataFrame, plot_type: str = 'histogram') -> str:
        """
        Generate visualizations based on data
        Returns path to saved image
        """
        plt.figure(figsize=(10, 6))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if not numeric_cols.empty:
            if plot_type == 'histogram':
                df[numeric_cols].hist()
                plt.tight_layout()
            elif plot_type == 'correlation':
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
            elif plot_type == 'pairplot' and len(numeric_cols) <= 5:
                sns.pairplot(df[numeric_cols])
            elif plot_type == 'boxplot':
                df[numeric_cols].boxplot()
            else:
                # Default to first numeric column histogram
                df[numeric_cols[0]].hist()
        else:
            # Handle non-numeric data
            if plot_type == 'countplot':
                for col in df.select_dtypes(include=['category', 'object']).columns:
                    sns.countplot(data=df, x=col)
                    plt.xticks(rotation=45)
            else:
                df[df.columns[0]].value_counts().plot(kind='bar')
        
        image_path = f"temp_plot_{plot_type}.png"
        plt.savefig(image_path)
        plt.close()
        return image_path

    @staticmethod
    def answer_question(df: pd.DataFrame, question: str) -> str:
        """
        Basic question answering based on data
        """
        try:
            # This is a simple implementation - in practice you'd use LLM here
            if "how many rows" in question.lower():
                return f"The dataset contains {len(df)} rows."
            elif "how many columns" in question.lower():
                return f"The dataset contains {len(df.columns)} columns."
            elif "columns" in question.lower() and "names" in question.lower():
                return f"The columns are: {', '.join(df.columns)}."
            elif "missing values" in question.lower():
                missing = df.isnull().sum().sum()
                return f"There are {missing} missing values in the dataset."
            else:
                return "I need more specific information to answer this question. Please ask about specific columns, statistics, or patterns in the data."
        except Exception as e:
            return f"Error answering question: {str(e)}"