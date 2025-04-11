import streamlit as st
from crew.analysis_crew import AnalysisCrew
import tempfile
import os
import time

# Set page config
st.set_page_config(
    page_title="AI Data Analysis Crew",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
            padding: 2rem;
        }
        .stTextArea textarea {
            min-height: 200px;
        }
        .report {
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stFileUploader {
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# App header
st.title("📊 Multi Agent AI Data Analysis Crew")
st.markdown("""
    Upload your CSV file and ask any data-related question. Our AI agents will analyze the data and provide insights.
""")

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state.uploaded_file = tmp_file.name
    
    # Display basic file info
    st.success(f"File uploaded successfully: {uploaded_file.name}")

# Question input
question = st.text_area(
    "Ask a question about your data (e.g., 'What are the key trends?', 'Are there any outliers?')",
    height=100
)

# Analyze button
if st.button("Analyze Data", disabled=not uploaded_file):
    if uploaded_file:
        with st.spinner("Assembling the AI analysis team..."):
            time.sleep(1)
            
        # Initialize the crew
        crew = AnalysisCrew()
        
        with st.spinner("Analyzing your data. This may take a few moments..."):
            try:
                # Execute the analysis
                result = crew.analyze_data(st.session_state.uploaded_file, question if question else None)
                
                # Store the result
                st.session_state.analysis_result = result
                
                # Display the result
                st.success("Analysis complete!")
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Analysis Report", "Raw Output"])
                
                with tab1:
                    st.subheader("Comprehensive Analysis Report")
                    st.markdown(f"<div class='report'>{st.session_state.analysis_result}</div>", unsafe_allow_html=True)
                
                with tab2:
                    st.subheader("Raw Analysis Output")
                    st.text(st.session_state.analysis_result)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
    else:
        st.warning("Please upload a CSV file first.")

# Clean up the temporary file when done
if 'uploaded_file' in st.session_state and st.session_state.uploaded_file:
    try:
        os.unlink(st.session_state.uploaded_file)
    except:
        pass

# Footer
st.markdown("---")
st.markdown("""
    *Built using CrewAI - A multi-agent framework for AI collaboration*
""")