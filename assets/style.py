
import streamlit as st

def load_custom_css():
    """Load custom CSS styling for the dashboard"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        }
        
        .alert-card {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(255,107,107,0.3);
        }
        
        .safe-card {
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,184,148,0.3);
        }
        
        .warning-card {
            background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(253,203,110,0.3);
        }
        
        .metric-box {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1e3c72;
            margin: 0.5rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .pathogen-high { 
            border-left-color: #d63031; 
            background: #ffeaa7; 
        }
        
        .pathogen-medium { 
            border-left-color: #e17055; 
            background: #fab1a0; 
        }
        
        .pathogen-low { 
            border-left-color: #00b894; 
            background: #b2f5ea; 
        }
        
        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px 10px 0 0;
            padding: 1rem 2rem;
            margin: 0 2px;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white !important;
        }
        
        /* Custom metric styling */
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        /* Loading spinner customization */
        .stSpinner > div {
            border-top-color: #1e3c72 !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Chart container styling */
        .js-plotly-plot {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Success/Error message styling */
        .stSuccess {
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
            border-radius: 8px;
        }
        
        .stError {
            background: linear-gradient(135deg, #d63031 0%, #e17055 100%);
            color: white;
            border-radius: 8px;
        }
        
        .stWarning {
            background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
            color: white;
            border-radius: 8px;
        }
        
        .stInfo {
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            border-radius: 8px;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        }
    </style>
    """, unsafe_allow_html=True)