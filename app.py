"""
Main Application File
Streamlit-based UI for YouTube Script Generator - Creators Buddy
"""

import streamlit as st
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# Import our modules
from config import Config
from transcript_processor import TranscriptProcessor, ProcessedTranscript
from script_generator import ScriptGenerator
from thumbnail_generator import build_thumbnail_prompt, generate_thumbnail, save_thumbnail
from thumbnail_overlay import render_text
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Creators Buddy",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject comprehensive dark mode/cinema theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #ffffff;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Glassmorphism cards */
    .creator-card, .metric-card, .script-section-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .creator-card:hover, .metric-card:hover, .script-section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(255, 0, 110, 0.2);
        border-color: rgba(255, 0, 110, 0.3);
    }
    
    /* Gradient buttons */
    .stButton>button {
        background: linear-gradient(135deg, #FF006E 0%, #8338EC 50%, #3A86FF 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 0, 110, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 0, 110, 0.5);
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        text-align: center;
        background: linear-gradient(135deg, #FF006E 0%, #8338EC 50%, #3A86FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Status cards */
    .status-card {
        background: rgba(131, 56, 236, 0.1);
        border: 2px solid rgba(131, 56, 236, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .status-card.ready {
        background: rgba(58, 134, 255, 0.1);
        border-color: rgba(58, 134, 255, 0.3);
    }
    
    .status-card.not-ready {
        background: rgba(255, 0, 110, 0.1);
        border-color: rgba(255, 0, 110, 0.3);
    }
    
    /* Script sections */
    .script-section-card {
        margin: 20px 0;
    }
    
    .script-section-title {
        color: #FF006E;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #3A86FF;
        font-weight: 700;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(10, 10, 10, 0.8);
    }
    
    /* Text inputs */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    /* Select boxes */
    .stSelectbox>div>div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    /* Copy button styling */
    .copy-btn {
        background: rgba(58, 134, 255, 0.2);
        border: 1px solid rgba(58, 134, 255, 0.4);
        border-radius: 8px;
        padding: 8px 16px;
        color: #3A86FF;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .copy-btn:hover {
        background: rgba(58, 134, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class YouTubeScriptGeneratorApp:
    """Main application class"""
    
    def __init__(self):
        self.initialize_app()
    
    def initialize_app(self):
        """Initialize the application state"""
        
        # Initialize session state
        if 'transcripts_loaded' not in st.session_state:
            st.session_state.transcripts_loaded = False
        if 'script_generator_trained' not in st.session_state:
            st.session_state.script_generator_trained = False
        if 'available_creators' not in st.session_state:
            st.session_state.available_creators = {}
        if 'training_summary' not in st.session_state:
            st.session_state.training_summary = {}
        if 'show_script_editor' not in st.session_state:
            st.session_state.show_script_editor = False
        if 'current_script' not in st.session_state:
            st.session_state.current_script = ""
        if 'current_metadata' not in st.session_state:
            st.session_state.current_metadata = {}
        if 'original_script' not in st.session_state:
            st.session_state.original_script = ""
        if 'content_format' not in st.session_state:
            st.session_state.content_format = "short-form"  # Default to short-form
        if 'selected_genre' not in st.session_state:
            st.session_state.selected_genre = None
        if 'available_genres' not in st.session_state:
            st.session_state.available_genres = []
        if 'use_personalized_training' not in st.session_state:
            st.session_state.use_personalized_training = False
        if 'creator_id' not in st.session_state:
            st.session_state.creator_id = ""
        
        # Initialize components
        try:
            Config.validate_config()
            self.data_dir = Config.DATA_DIR
            # Cache available genres
            from transcript_processor import TranscriptProcessor
            st.session_state.available_genres = TranscriptProcessor.get_available_genres()
        except Exception as e:
            st.error(f"Configuration error: {e}")
            st.stop()
    
    def render_dashboard(self):
        """Render the dashboard with system status and initialization"""
        
        st.header("📊 System Dashboard")
        
        # System Status Cards
        col1, col2 = st.columns(2)
        
        with col1:
            status_class = "ready" if st.session_state.transcripts_loaded else "not-ready"
            status_text = "✓ Ready" if st.session_state.transcripts_loaded else "✗ Not Loaded"
            st.markdown(f"""
            <div class="status-card {status_class}">
                <h3 style="margin: 0; color: {'#3A86FF' if st.session_state.transcripts_loaded else '#FF006E'};">
                    Transcripts Status
                </h3>
                <p style="font-size: 1.5rem; margin: 10px 0;">{status_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_class = "ready" if st.session_state.script_generator_trained else "not-ready"
            status_text = "✓ Trained" if st.session_state.script_generator_trained else "✗ Not Trained"
            st.markdown(f"""
            <div class="status-card {status_class}">
                <h3 style="margin: 0; color: {'#3A86FF' if st.session_state.script_generator_trained else '#FF006E'};">
                    Model Status
                </h3>
                <p style="font-size: 1.5rem; margin: 10px 0;">{status_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Initialize System Section
        if not st.session_state.transcripts_loaded or not st.session_state.script_generator_trained:
            st.markdown("---")
            st.subheader("🚀 Initialize System")
            
            # Content format selection
            col1, col2 = st.columns(2)
            with col1:
                format_choice = st.radio(
                    "Content Format",
                    options=["Short-form", "Long-form"],
                    index=0 if st.session_state.content_format == "short-form" else 1,
                    horizontal=True,
                    key="dashboard_format"
                )
                st.session_state.content_format = "short-form" if format_choice == "Short-form" else "long-form"
            
            with col2:
                if st.session_state.content_format == "long-form":
                    if st.session_state.available_genres:
                        genre = st.selectbox(
                            "Select Genre",
                            options=st.session_state.available_genres,
                            key="dashboard_genre"
                        )
                        st.session_state.selected_genre = genre
                    else:
                        st.warning("No genres found. Check Data/LONG-FORM directory.")
            
            # Initialize button
            if st.button("🎯 Initialize System", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Load transcripts
                    if not st.session_state.transcripts_loaded:
                        status_text.text("📂 Loading transcripts...")
                        progress_bar.progress(0.3)
                        self.load_transcripts()
                        st.session_state.transcripts_loaded = True
                        progress_bar.progress(0.6)
                    
                    # Step 2: Train generator
                    if not st.session_state.script_generator_trained:
                        status_text.text("🧠 Training model...")
                        progress_bar.progress(0.7)
                        self.train_generator()
                        st.session_state.script_generator_trained = True
                        progress_bar.progress(1.0)
                    
                    status_text.text("✓ System initialized successfully!")
                    st.toast("🎉 System initialized successfully!", icon="✅")
                    time.sleep(0.5)
                    st.rerun()
                    
                except Exception as e:
                    progress_bar.progress(0)
                    status_text.text("❌ Initialization failed")
                    st.error(f"Error: {e}")
                    st.toast(f"❌ Error: {e}", icon="❌")
        else:
            # System is ready - show stats
            st.markdown("---")
            st.subheader("📈 Quick Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                creator_count = len(st.session_state.available_creators)
                st.metric("Creators Analyzed", creator_count)
            
            with col2:
                transcript_count = st.session_state.training_summary.get('total_transcripts', 0)
                st.metric("Transcripts Processed", transcript_count)
            
            with col3:
                if creator_count > 0 and transcript_count > 0:
                    avg_videos = transcript_count / creator_count
                    st.metric("Avg Videos/Creator", f"{avg_videos:.1f}")
                else:
                    st.metric("Avg Videos/Creator", "N/A")
            
            with col4:
                format_display = "Short-form" if st.session_state.content_format == "short-form" else "Long-form"
                st.metric("Content Format", format_display)
            
            # Quick actions
            st.markdown("---")
            st.subheader("⚡ Quick Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Reload Data", use_container_width=True):
                    try:
                        self.load_transcripts()
                        st.toast("✓ Data reloaded successfully!", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.toast(f"❌ Error: {e}", icon="❌")
            
            with col2:
                if st.button("🧠 Retrain Model", use_container_width=True):
                    try:
                        self.train_generator()
                        st.toast("✓ Model retrained successfully!", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.toast(f"❌ Error: {e}", icon="❌")
            
            with col3:
                if st.button("📊 View Analysis", use_container_width=True):
                    st.switch_page("Analysis")
    
    def run(self):
        """Run the main application"""
        
        # Header
        st.markdown('<h1 class="main-header">🎬 Creators Buddy</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.7); font-size: 1.1rem; margin-bottom: 2rem;">Generate Authentic Hinglish YouTube Scripts with AI</p>', unsafe_allow_html=True)
        
        # Sidebar - Global Settings
        with st.sidebar:
            self.render_sidebar()
        
        # Main navigation using option_menu
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Script Lab", "Content Planner", "Analysis", "Creator DNA", "Thumbnail Studio", "Analytics"],
            icons=["speedometer2", "file-text", "calendar3", "graph-up", "person-badge", "image", "bar-chart"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "rgba(0,0,0,0.5)"},
                "icon": {"color": "#FF006E", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                    "color": "#ffffff",
                    "background-color": "rgba(255,255,255,0.05)",
                    "border-radius": "8px",
                    "padding": "10px",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, #FF006E 0%, #8338EC 100%)",
                    "color": "#ffffff",
                },
            }
        )
        
        # Route to appropriate page
        if selected == "Dashboard":
            self.render_dashboard()
        elif selected == "Script Lab":
            self.render_script_lab()
        elif selected == "Content Planner":
            self.render_content_planner()
        elif selected == "Analysis":
            self.render_analysis_tab()
        elif selected == "Creator DNA":
            self.render_creator_styles_tab()
        elif selected == "Thumbnail Studio":
            self.render_thumbnail_studio()
        elif selected == "Analytics":
            self.render_analytics_dashboard()
    
    def render_sidebar(self):
        """Render the sidebar with global settings"""
        
        st.header("⚙️ Global Settings")
        
        # API Status
        st.subheader("🔑 API Status")
        api_key_status = "✓ Configured" if Config.GEMINI_API_KEY else "✗ Missing"
        st.markdown(f"**Gemini API:** {api_key_status}")
        
        if not Config.GEMINI_API_KEY:
            st.warning("Please set GEMINI_API_KEY in your .env file")
        
        # System Info
        st.subheader("ℹ️ System Info")
        st.markdown(f"**Content Format:** {st.session_state.content_format.replace('-', ' ').title()}")
        if st.session_state.content_format == "long-form" and st.session_state.selected_genre:
            st.markdown(f"**Genre:** {st.session_state.selected_genre}")
        
        # Quick Stats (if loaded)
        if st.session_state.transcripts_loaded:
            st.markdown("---")
            st.subheader("📊 Quick Stats")
            creator_count = len(st.session_state.available_creators)
            transcript_count = st.session_state.training_summary.get('total_transcripts', 0)
            st.metric("Creators", creator_count)
            st.metric("Transcripts", transcript_count)
        
        # About Section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **Creators Buddy** v1.0
        
        AI-powered YouTube script generator for authentic Hinglish content.
        
        Built with Streamlit & Gemini AI
        """)
    
    def load_transcripts(self):
        """Load and process transcript data based on selected format"""
        
        content_format = st.session_state.content_format
        genre = st.session_state.selected_genre if content_format == "long-form" else None
        
        # Load user scripts if personalized training is enabled
        if st.session_state.use_personalized_training and st.session_state.creator_id:
            processor = TranscriptProcessor(content_format=content_format, genre=genre)
            processed_transcripts = processor.load_user_scripts(st.session_state.creator_id)
            if not processed_transcripts:
                raise ValueError(f"No user scripts found for creator ID: {st.session_state.creator_id}")
        else:
            # Load from Data-2 structure
            processor = TranscriptProcessor(content_format=content_format, genre=genre)
            processed_transcripts = processor.load_all_transcripts()
            
            if not processed_transcripts:
                raise ValueError(f"No transcripts could be loaded for {content_format}" + (f" genre: {genre}" if genre else ""))
        
        # Store in session state
        st.session_state.processed_transcripts = processed_transcripts
        st.session_state.creator_summaries = processor.get_creator_summary()
        st.session_state.available_creators = processor.get_creator_summary()
        
        print(f"Loaded {len(processed_transcripts)} transcripts for {content_format}")
    
    def train_generator(self):
        """Train the script generator"""
        
        processed_transcripts = st.session_state.processed_transcripts
        
        generator = ScriptGenerator()
        training_summary = generator.train_on_transcripts(processed_transcripts)
        
        # Store in session state
        st.session_state.script_generator = generator
        st.session_state.training_summary = training_summary
        
        print(f"Training completed: {training_summary}")
    
    def render_script_lab(self):
        """Render the script generation interface with enhanced UI"""
        
        if not st.session_state.transcripts_loaded or not st.session_state.script_generator_trained:
            st.warning("⚠️ Please initialize the system from the Dashboard first")
            if st.button("Go to Dashboard"):
                st.switch_page("Dashboard")
            return
        
        st.header("🚀 Script Lab")
        st.markdown("Generate authentic Hinglish YouTube scripts with AI")
        
        # Content Format Selection
        st.subheader("📋 Content Format")
        content_format = st.radio(
            "Select Content Format",
            options=["Short-form", "Long-form"],
            index=0 if st.session_state.content_format == "short-form" else 1,
            horizontal=True,
            help="Short-form: 10-120 seconds | Long-form: 5+ minutes with genre selection"
        )
        content_format_lower = "short-form" if content_format == "Short-form" else "long-form"
        st.session_state.content_format = content_format_lower
        
        # Generation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Content Parameters")
            
            # Topic input
            topic = st.text_input(
                "Video Topic",
                placeholder="e.g., iPhone 15 Pro Max Review",
                help="Enter the main topic or product for the video"
            )
            
            # Length selection based on format
            if content_format_lower == "short-form":
                length_seconds = st.slider(
                    "Video Length (seconds)",
                    min_value=10,
                    max_value=120,
                    value=60,
                    step=5,
                    help="Target video duration in seconds (10-120 seconds)"
                )
                length_minutes = None
                
                # Show estimated word count
                estimated_words = int(length_seconds * Config.SHORT_FORM_WPS)
                st.info(f"📊 Estimated word count: ~{estimated_words} words")
            else:
                # Long-form: Genre selection
                if st.session_state.available_genres:
                    selected_genre = st.selectbox(
                        "Genre",
                        options=st.session_state.available_genres,
                        help="Select the genre/niche for long-form content"
                    )
                    st.session_state.selected_genre = selected_genre
                else:
                    st.warning("No genres available. Please check Data-2/LONG-FORM/Genres directory.")
                    selected_genre = None
                
                length_minutes = st.selectbox(
                    "Video Length (minutes)",
                    options=[5, 8, 10, 12, 15, 18, 20, 25],
                    index=2,  # Default to 10 minutes
                    help="Target video duration in minutes"
                )
                length_seconds = None
                
                # Show estimated word count
                estimated_words = int(length_minutes * Config.SPEECH_WPM)
                st.info(f"📊 Estimated word count: ~{estimated_words} words")
            
            # Content type
            content_type = st.selectbox(
                "Content Type",
                options=['review', 'comparison', 'unboxing', 'tutorial', 'general'],
                help="Type of YouTube content"
            )
            
            # Outline input
            outline = st.text_area(
                "Script Outline (Optional)",
                placeholder="Enter your script outline here. The model will improve and enhance it while keeping the same flow...",
                height=100,
                help="Provide an outline that will be enhanced and used as the foundation for the script"
            )
            
            # Facts and Information input
            facts = st.text_area(
                "Facts & Information",
                placeholder="Enter key facts, information, or data points around which the script will be generated...",
                height=100,
                help="Core information and facts that should be included in the script"
            )
        
        with col2:
            st.subheader("🎭 Style Parameters")
            
            # Personalized Training Toggle
            use_personalized = st.checkbox(
                "Use Personalized Training",
                value=st.session_state.use_personalized_training,
                help="Enable to use only your uploaded scripts for training"
            )
            st.session_state.use_personalized_training = use_personalized
            
            if use_personalized:
                creator_id = st.text_input(
                    "Creator ID",
                    value=st.session_state.creator_id,
                    placeholder="Enter your creator ID",
                    help="Your unique creator identifier for personalized training"
                )
                st.session_state.creator_id = creator_id
            
            # Creator style
            if not use_personalized:
                creator_options = ["Auto-select (best match)"] + list(st.session_state.available_creators.keys())
                creator_style = st.selectbox(
                    "Creator Style",
                    options=creator_options,
                    help="Choose which creators style to replicate"
                )
            else:
                creator_style = None
            
            # Tone
            tone = st.selectbox(
                "Script Tone",
                options=Config.VALID_TONES,
                index=0,
                help="Overall tone and energy level"
            )
            
            # Target audience
            target_audience = st.selectbox(
                "Target Audience",
                options=Config.VALID_AUDIENCES,
                index=0,
                help="Primary audience for the script"
            )
            
            # Language mix preference - UPDATED: English instead of English Heavy
            language_mix = st.select_slider(
                "Language Mix Preference",
                options=['Hindi Heavy', 'Balanced', 'English'],
                value='Balanced',
                help="Preferred language: Hindi Heavy (mostly Hindi), Balanced (Hinglish mix), English (English only)"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            temperature = st.slider("Creativity Level", 0.1, 1.0, 0.7, 0.1)
            
            force_hinglish_ascii = st.checkbox(
                "Force Hinglish (Latin script only)",
                value=True,
                disabled=(language_mix == "English"),
                help="When enabled, the model will write Hindi using Latin letters (e.g., 'aap kya kar rahe ho') to avoid Devanagari spacing issues. Disabled for English mode."
            )
        
        # Generate button
        if st.button("🎬 Generate Script", type="primary", use_container_width=True):
            if not topic:
                st.toast("❌ Please enter a video topic", icon="❌")
                return
            
            if content_format_lower == "long-form":
                if not st.session_state.selected_genre:
                    st.toast("❌ Please select a genre for long-form content", icon="❌")
                    return
                selected_genre = st.session_state.selected_genre
            else:
                selected_genre = None
            
            if use_personalized and not st.session_state.creator_id:
                st.toast("❌ Please enter a Creator ID when using personalized training", icon="❌")
                return
            
            # Prepare parameters
            params = {
                'topic': topic,
                'length_minutes': length_minutes,
                'length_seconds': length_seconds,
                'tone': tone,
                'target_audience': target_audience,
                'content_type': content_type,
                'content_format': content_format_lower,
                'creator_style': creator_style if creator_style and creator_style != "Auto-select (best match)" else None,
                'outline': outline if outline else None,
                'facts': facts if facts else None,
                'language_mix': language_mix,
                'force_hinglish_ascii': force_hinglish_ascii if language_mix != "English" else False
            }
            
            # Show generation progress
            with st.spinner("🔄 Generating your script... This may take a moment."):
                try:
                    # Generate script
                    generator = st.session_state.script_generator
                    
                    # Update generation config if advanced options changed
                    generator.generation_config["temperature"] = temperature
                    
                    result = generator.generate_script(**params)
                    
                    # Display results
                    if result['success']:
                        st.toast("🎉 Script generated successfully!", icon="✅")
                        self.display_generated_script(result)
                    else:
                        st.toast(f"❌ Generation failed: {result.get('error', 'Unknown error')}", icon="❌")
                        st.error(f"Generation failed: {result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    st.toast(f"❌ Error: {str(e)}", icon="❌")
                    st.error(f"Error generating script: {e}")
        
        # Examples section
        with st.expander("💡 Example Prompts"):
            if content_format_lower == "short-form":
                st.markdown("""
                **Short-form topic examples:**
                
                - `Quick Tips: How to Save Battery Life`
                - `5 Second Recipe Hack`
                - `One-Liner Joke Setup`
                - `Quick Product Demo`
                - `Funny Life Hack`
                
                **Pro tip:** Keep topics concise and punchy for short-form content!
                """)
            else:
                st.markdown("""
                **Long-form topic examples:**
                
                - `Complete Guide to Smartphone Photography`
                - `In-depth Review: Samsung Galaxy S24 Ultra`
                - `How to Build Your First Gaming PC`
                - `Understanding Machine Learning Basics`
                - `Top 10 Productivity Apps Explained`
                
                **Pro tip:** Be specific about what you want to cover for better results!
                """)
        
        # Safety guidelines
        with st.expander("⚠️ Content Guidelines"):
            st.markdown("""
            **To avoid content blocking, please:**
            
            - Use clear, educational topics
            - Avoid controversial or sensitive subjects
            - Focus on informative and helpful content
            - Keep content family-friendly and professional
            - Be specific about features and information
            
            **If generation fails:** Try rephrasing your topic or using more neutral language.
            """)
        
        # Script Editor (if script exists or editor is open)
        if st.session_state.show_script_editor or st.session_state.current_script:
            st.markdown("---")
            self.render_script_editor()
    
    def _parse_script_sections(self, script: str) -> Dict[str, str]:
        """Parse script into sections (Hook, Intro, Body, CTA, Outro)"""
        sections = {
            'hook': '',
            'intro': '',
            'body': '',
            'cta': '',
            'outro': ''
        }
        
        # Try to find section markers
        lines = script.split('\n')
        current_section = 'body'
        
        for line in lines:
            line_upper = line.upper().strip()
            if '[HOOK]' in line_upper or line_upper.startswith('HOOK'):
                current_section = 'hook'
                continue
            elif '[INTRO]' in line_upper or line_upper.startswith('INTRO'):
                current_section = 'intro'
                continue
            elif '[CTA]' in line_upper or 'CALL TO ACTION' in line_upper:
                current_section = 'cta'
                continue
            elif '[OUTRO]' in line_upper or line_upper.startswith('OUTRO'):
                current_section = 'outro'
                continue
            elif '[MAIN' in line_upper or '[BODY]' in line_upper:
                current_section = 'body'
                continue
            
            if line.strip() and not line.strip().startswith('['):
                sections[current_section] += line + '\n'
        
        # If no sections found, put everything in body
        if not any(sections.values()):
            sections['body'] = script
        
        return sections
    
    def _copy_to_clipboard_button(self, text: str, label: str = "Copy"):
        """Create a copy button - text is already in st.code which has built-in copy"""
        # st.code already has a copy button, so we just show a visual indicator
        pass
    
    def display_generated_script(self, result: Dict):
        """Display the generated script with enhanced formatting"""
        
        st.markdown("---")
        st.header("✨ Generated Script")
        
        metadata = result['metadata']
        script = result['script']
        
        # Store script in session state for editing
        st.session_state.current_script = script
        st.session_state.current_metadata = metadata
        st.session_state.original_script = script
        
        # Metadata cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⏱️ Duration", f"{metadata.get('length_minutes', metadata.get('length_seconds', 0) / 60):.1f} min")
        
        with col2:
            st.metric("📝 Words", f"{metadata['estimated_word_count']:,}")
        
        with col3:
            st.metric("🎭 Tone", metadata['tone_used'].replace('_', ' ').title())
        
        with col4:
            st.metric("⚡ Gen Time", f"{metadata['generation_time_seconds']:.1f}s")
        
        # Floating action bar
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📋 Copy Script", key="copy_script_btn", use_container_width=True):
                st.toast("✓ Script copied to clipboard!", icon="✅")
                # Note: Actual clipboard copy requires JavaScript - handled in display
        
        with col2:
            if st.button("💾 Save Script", key="save_script_btn", use_container_width=True):
                self.save_script_to_file(script, metadata)
                st.toast("✓ Script saved successfully!", icon="✅")
        
        with col3:
            if st.button("📊 Analyze", key="analyze_script_btn", use_container_width=True):
                self.analyze_script(script, metadata)
        
        with col4:
            if st.button("✏️ Edit", key="edit_script_btn", use_container_width=True):
                st.session_state.show_script_editor = True
                st.rerun()
        
        with col5:
            if st.button("📥 Download", key="download_script_btn", use_container_width=True):
                self._download_script(script, metadata)
        
        # Parse and display script sections
        sections = self._parse_script_sections(script)
        
        st.markdown("---")
        st.subheader("📑 Script Sections")
        
        # Hook Section
        if sections['hook']:
            st.markdown("""
            <div class="script-section-card">
                <div class="script-section-title">🎣 Hook</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"```\n{sections['hook'].strip()}\n```")
            self._copy_to_clipboard_button(sections['hook'].strip(), "Copy Hook")
        
        # Intro Section
        if sections['intro']:
            st.markdown("""
            <div class="script-section-card">
                <div class="script-section-title">👋 Introduction</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"```\n{sections['intro'].strip()}\n```")
            self._copy_to_clipboard_button(sections['intro'].strip(), "Copy Intro")
        
        # Body Section
        if sections['body']:
            st.markdown("""
            <div class="script-section-card">
                <div class="script-section-title">📝 Main Content</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"```\n{sections['body'].strip()}\n```")
            self._copy_to_clipboard_button(sections['body'].strip(), "Copy Body")
        
        # CTA Section
        if sections['cta']:
            st.markdown("""
            <div class="script-section-card">
                <div class="script-section-title">📢 Call to Action</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"```\n{sections['cta'].strip()}\n```")
            self._copy_to_clipboard_button(sections['cta'].strip(), "Copy CTA")
        
        # Outro Section
        if sections['outro']:
            st.markdown("""
            <div class="script-section-card">
                <div class="script-section-title">👋 Outro</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"```\n{sections['outro'].strip()}\n```")
            self._copy_to_clipboard_button(sections['outro'].strip(), "Copy Outro")
        
        # Full script view
        with st.expander("📄 View Full Script", expanded=False):
            st.code(script, language=None)
            self._copy_to_clipboard_button(script, "Copy Full Script")
        
        # Timing breakdown
        with st.expander("⏰ Timing Breakdown"):
            timing_suggestions = result['timing_suggestions']
            
            st.markdown(f"""
            - **Hook:** {timing_suggestions['hook_duration']}
            - **Intro:** {timing_suggestions['intro_duration']}
            - **CTA:** Around {timing_suggestions['cta_timing']}
            - **Outro:** From {timing_suggestions['outro_timing']}
            """)
        
        # Applied patterns
        if 'creator_patterns_applied' in result and result['creator_patterns_applied']:
            with st.expander("🎯 Applied Creator Patterns"):
                patterns = result['creator_patterns_applied']
                
                cols = st.columns(2)
                with cols[0]:
                    if patterns.get('hinglish_expressions'):
                        st.write("**Hinglish Expressions:**", ", ".join(patterns['hinglish_expressions']))
                
                with cols[1]:
                    if patterns.get('engagement_phrases'):
                        st.write("**Engagement Phrases:**", ", ".join(patterns['engagement_phrases']))
        
        # Structured output view
        if 'structured' in result and result['structured']:
            structured_data = result['structured']
            
            with st.expander("📋 Structured Script View", expanded=False):
                # Display structured sections
                st.subheader("📑 Script Sections")
                
                tabs = st.tabs(["Overview", "Hook", "Intro", "Main Content", "CTA", "Outro", "JSON Export", "XML Export", "Markdown Export"])
                
                with tabs[0]:  # Overview
                    st.markdown("### Script Overview")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Duration", f"{structured_data['total_duration_seconds']}s")
                    with col2:
                        st.metric("Total Words", structured_data['total_word_count'])
                    with col3:
                        st.metric("Sections", len([s for s in structured_data['sections'].values() if s['text']]))
                    
                    st.markdown("### Section Summary")
                    for section_name, section_data in structured_data['sections'].items():
                        if section_data['text']:
                            section_title = section_name.replace('_', ' ').title()
                            st.markdown(f"**{section_title}**: {section_data['word_count']} words | "
                                      f"{section_data['start_time']}s - {section_data['end_time']}s "
                                      f"({section_data['duration']}s)")
                
                # Section tabs
                section_mapping = {
                    1: ('hook', 'Hook'),
                    2: ('intro', 'Introduction'),
                    3: ('main_content', 'Main Content'),
                    4: ('cta', 'Call to Action'),
                    5: ('outro', 'Outro')
                }
                
                for tab_idx, (section_key, section_title) in section_mapping.items():
                    with tabs[tab_idx]:
                        section_data = structured_data['sections'][section_key]
                        if section_data['text']:
                            st.markdown(f"### {section_title} Section")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Timing:** {section_data['start_time']}s - {section_data['end_time']}s")
                            with col2:
                                st.markdown(f"**Duration:** {section_data['duration']}s")
                            st.markdown(f"**Word Count:** {section_data['word_count']}")
                            st.markdown("---")
                            st.markdown(section_data['text'])
                        else:
                            st.info(f"No {section_title.lower()} section found in this script.")
                
                # Export tabs
                with tabs[6]:  # JSON Export
                    st.markdown("### Export as JSON")
                    json_output = json.dumps(structured_data, ensure_ascii=False, indent=2)
                    st.code(json_output, language='json')
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_output,
                        file_name=f"script_{structured_data['script_id']}.json",
                        mime="application/json"
                    )
                
                with tabs[7]:  # XML Export
                    st.markdown("### Export as XML")
                    from script_generator import ScriptGenerator
                    generator = ScriptGenerator()
                    xml_output = generator.export_structured_script(structured_data, format='xml')
                    st.code(xml_output, language='xml')
                    st.download_button(
                        label="📥 Download XML",
                        data=xml_output,
                        file_name=f"script_{structured_data['script_id']}.xml",
                        mime="application/xml"
                    )
                
                with tabs[8]:  # Markdown Export
                    st.markdown("### Export as Markdown")
                    from script_generator import ScriptGenerator
                    generator = ScriptGenerator()
                    md_output = generator.export_structured_script(structured_data, format='markdown')
                    st.code(md_output, language='markdown')
                    st.download_button(
                        label="📥 Download Markdown",
                        data=md_output,
                        file_name=f"script_{structured_data['script_id']}.md",
                        mime="text/markdown"
                    )
    
    def render_script_editor(self):
        """Render the script editor interface"""
        
        if 'current_script' not in st.session_state:
            st.warning("No script available for editing. Please generate a script first.")
            return
        
        st.header("✏️ Script Editor")
        st.markdown("Edit your generated script and save changes in real-time.")
        
        # Editor interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Text area for editing
            edited_script = st.text_area(
                "Edit Script",
                value=st.session_state.current_script,
                height=400,
                help="Make your changes to the script here"
            )
            
            # Update session state with edited script
            st.session_state.current_script = edited_script
        
        with col2:
            st.subheader("📊 Live Stats")
            
            # Calculate live metrics
            word_count = len(edited_script.split())
            char_count = len(edited_script)
            estimated_minutes = word_count / 150  # Average speaking pace
            
            st.metric("Word Count", f"{word_count:,}")
            st.metric("Characters", f"{char_count:,}")
            from config import Config
            estimated_minutes = word_count / Config.SPEECH_WPM
            st.metric("Est. Duration", f"{estimated_minutes:.1f} min @ {Config.SPEECH_WPM} wpm")
            
            # Language mix analysis
            if edited_script:
                from script_validator import ScriptValidator
                validator = ScriptValidator()
                language_mix = validator._analyze_language_mix(edited_script)
                
                st.markdown("**Language Mix:**")
                st.markdown(f"Hindi: {language_mix['hindi_ratio']:.1%}")
                st.markdown(f"English: {language_mix['english_ratio']:.1%}")
                st.markdown(f"Mixed: {language_mix['mixed_ratio']:.1%}")
        
        # Action buttons for editor
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Changes", key="save_edited_script"):
                self.save_script_to_file(edited_script, st.session_state.current_metadata)
        
        with col2:
            if st.button("📊 Analyze Edited", key="analyze_edited_script"):
                self.analyze_script(edited_script, st.session_state.current_metadata)
        
        with col3:
            if st.button("🔄 Reset to Original", key="reset_script"):
                st.session_state.current_script = st.session_state.original_script
                st.rerun()
        
        with col4:
            if st.button("❌ Close Editor", key="close_editor"):
                st.session_state.show_script_editor = False
                st.rerun()
        
        # Show preview of edited script
        with st.expander("👀 Preview Edited Script"):
            st.markdown(edited_script.replace('\n', '\n\n'))
    
    def render_analysis_tab(self):
        """Render the data analysis tab"""
        
        if not st.session_state.transcripts_loaded:
            st.info("👆 Please load transcripts first to view analysis")
            return
        
        st.header("📊 Transcript Data Analysis")
        
        transcripts = st.session_state.processed_transcripts
        creator_summaries = st.session_state.creator_summaries
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_transcripts = len(transcripts)
            st.metric("Total Transcripts", total_transcripts)
        
        with col2:
            creator_count = len(creator_summaries)
            st.metric("Creators", creator_count)
        
        with col3:
            avg_duration = sum(t.metadata.duration for t in transcripts) / len(transcripts) / 60
            st.metric("Avg Duration", f"{avg_duration:.1f} min")
        
        with col4:
            total_words = sum(t.metadata.word_count for t in transcripts)
            st.metric("Total Words", f"{total_words:,}")
        
        # Creator breakdown
        st.subheader("🏆 Creator Breakdown")
        
        for creator, summary in creator_summaries.items():
            with st.container():
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="creator-card">
                        <h4>🎭 {creator}</h4>
                        <p><strong>Videos:</strong> {summary['video_count']} | 
                        <strong>Total Duration:</strong> {summary['total_duration'] // 60} min</p>
                        <p><strong>Language Mix:</strong> {summary['language_mix']['hindi_ratio']:.1%} Hindi, 
                        {summary['language_mix']['english_ratio']:.1%} English</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Avg Words/Video", f"{summary['total_words'] // summary['video_count']:,}")
        
        # Language analysis
        st.subheader("🗣️ Language Analysis")
        
        # Overall language distribution
        hindi_total = sum(t.language_breakdown['hindi'] for t in transcripts)
        english_total = sum(t.language_breakdown['english'] for t in transcripts)
        mixed_total = sum(t.language_breakdown['mixed'] for t in transcripts)
        total_all = hindi_total + english_total + mixed_total
        
        if total_all > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                hindi_percent = (hindi_total / total_all) * 100
                st.metric("Hindi Words", f"{hindi_percent:.1f}%")
            
            with col2:
                english_percent = (english_total / total_all) * 100
                st.metric("English Words", f"{english_percent:.1f}%")
            
            with col3:
                mixed_percent = (mixed_total / total_all) * 100
                st.metric("Mixed Words", f"{mixed_percent:.1f}%")
        
        # Tone analysis
        st.subheader("🎭 Tone Analysis")
        
        avg_enthusiasm = sum(t.tone_markers['enthusiasm'] for t in transcripts) / len(transcripts)
        avg_technical = sum(t.tone_markers['technical_depth'] for t in transcripts) / len(transcripts)
        avg_friendly = sum(t.tone_markers['friendliness'] for t in transcripts) / len(transcripts)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Enthusiasm", f"{avg_enthusiasm:.1f}/10")
        
        with col2:
            st.metric("Avg Technical Detail", f"{avg_technical:.1f}/10")
        
        with col3:
            st.metric("Avg Friendliness", f"{avg_friendly:.1f}/10")
    
    def render_creator_styles_tab(self):
        """Render creator styles analysis tab"""
        
        if not st.session_state.transcripts_loaded:
            st.info("👆 Please load transcripts first to view creator styles")
            return
        
        st.header("🎭 Creator Style Analysis")
        
        creator_summaries = st.session_state.creator_summaries
        
        for creator, summary in creator_summaries.items():
            st.subheader(f"📺 {creator}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Tone Profile**")
                
                enthusiasm = summary['avg_tone']['enthusiasm']
                technical = summary['avg_tone']['technical_depth']
                friendly = summary['avg_tone']['friendliness']
                
                st.metric("Enthusiasm", f"{enthusiasm:.1f}/10")
                st.metric("Technical Depth", f"{technical:.1f}/10")
                st.metric("Friendliness", f"{friendly:.1f}/10")
            
            with col2:
                st.markdown("**🗣️ Language Preferences**")
                
                hindi_ratio = summary['language_mix']['hindi_ratio']
                english_ratio = summary['language_mix']['english_ratio']
                mixed_ratio = summary['language_mix']['mixed_ratio']
                
                st.metric("Hindi Ratio", f"{hindi_ratio:.1%}")
                st.metric("English Ratio", f"{english_ratio:.1%}")
                st.metric("Mixed Ratio", f"{mixed_ratio:.1%}")
            
            # Common keywords
            if summary['common_keywords']:
                st.markdown("**🔑 Common Keywords**")
                keywords_str = ", ".join(summary['common_keywords'][:10])
                st.markdown(f"*{keywords_str}*")
            
            # Style markers
            if summary['style_markers']:
                st.markdown("**🎯 Style Markers**")
                markers_str = ", ".join(summary['style_markers'])
                st.markdown(f"*{markers_str}*")
            
            st.markdown("---")
    
    def render_settings_tab(self):
        """Render settings and configuration tab"""
        
        st.header("⚙️ Settings & Configuration")
        
        st.subheader("🔧 Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **📂 Data Directory:** `{Config.DATA_DIR}`  
            **🧠 Max Script Length:** {Config.MAX_SCRIPT_LENGTH_CHARS:,} characters  
            **📊 Transcripts to Load:** {Config.NUM_TRANSCRIPTS_TO_LOAD}  
            """)
        
        with col2:
            st.markdown(f"""
            **🌡️ Default Temperature:** 0.7  
            **🎭 Valid Tones:** {len(Config.VALID_TONES)} options  
            **👥 Valid Audiences:** {len(Config.VALID_AUDIENCES)} options  
            """)
        
        st.subheader("📋 Available Tones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            for tone in Config.VALID_TONES[:4]:
                st.markdown(f"• {tone}")
        
        with col2:
            for tone in Config.VALID_TONES[4:]:
                st.markdown(f"• {tone}")
        
        st.subheader("👥 Target Audiences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            for audience in Config.VALID_AUDIENCES[:4]:
                st.markdown(f"• {audience}")
        
        with col2:
            for audience in Config.VALID_AUDIENCES[4:]:
                st.markdown(f"• {audience}")
        
        # Model information
        st.subheader("🤖 Model Information")
        
        st.markdown("""
        **Model Used:** Google Gemini 2.0 Flash  
        **Training Approach:** Context-aware prompting with transcript analysis  
        **Language Mix:** Automatic detection and replication  
        **Style Adaptation:** Creator-specific patterns and preferences  
        """)
        st.markdown(f"**HF Image Model:** `{Config.HF_MODEL_ID}` (fallback: `{Config.HF_MODEL_FALLBACK}`)  ")
        
        # Export/Import options
        st.subheader("💾 Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Training Data"):
                self.export_training_data()
        
        with col2:
            uploaded_file = st.file_uploader("📥 Import Training Data", type=['json'])
            if uploaded_file:
                st.info("Import functionality coming soon!")
    
    def _download_script(self, script: str, metadata: Dict):
        """Create download button for script"""
        content = f"""TOPIC: {metadata['topic']}
DURATION: {metadata.get('length_minutes', metadata.get('length_seconds', 0) / 60):.1f} minutes
TONE: {metadata['tone_used']}
TARGET AUDIENCE: {metadata['target_audience']}
CONTENT TYPE: {metadata.get('content_type', 'general')}
CREATED: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

{script}
"""
        st.download_button(
            label="📥 Download as TXT",
            data=content,
            file_name=f"script_{int(time.time())}.txt",
            mime="text/plain"
        )
    
    def save_script_to_file(self, script: str, metadata: Dict):
        """Save generated script to file"""
        
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = int(time.time())
        filename = f"script_{timestamp}.txt"
        filepath = output_dir / filename
        
        # Save script with metadata
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"TOPIC: {metadata['topic']}\n")
            if metadata.get('length_seconds'):
                f.write(f"DURATION: {metadata['length_seconds']} seconds\n")
            elif metadata.get('length_minutes'):
                f.write(f"DURATION: {metadata['length_minutes']} minutes\n")
            if metadata.get('content_format'):
                f.write(f"FORMAT: {metadata['content_format']}\n")
            f.write(f"TONE: {metadata['tone_used']}\n")
            f.write(f"TARGET AUDIENCE: {metadata['target_audience']}\n")
            f.write(f"CONTENT TYPE: {metadata['content_type']}\n")
            f.write(f"CREATED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(script)
        
        st.toast(f"✓ Script saved to {filepath}", icon="✅")
    
    def analyze_script(self, script: str, metadata: Dict = None):
        """Analyze the generated script using the validator"""
        
        if not script:
            st.error("No script provided for analysis")
            return
        
        try:
            from script_validator import ScriptValidator
            
            # Get metadata or use defaults
            if metadata is None:
                metadata = st.session_state.get('current_metadata', {})
            
            target_length = metadata.get('length_minutes') or metadata.get('length_seconds', 0) / 60.0 or 10
            # Ensure target_length is a valid number
            if target_length is None or target_length <= 0:
                target_length = 10  # Default to 10 minutes
            
            target_tone = metadata.get('tone_used', 'friendly_and_informative')
            target_audience = metadata.get('target_audience', 'general_audience')
            content_type = metadata.get('content_type', 'general')
            creator_style = metadata.get('creator_style', None)
            
            # Perform validation
            validator = ScriptValidator()
            validation_result = validator.validate_script(
                script=script,
                target_length_minutes=int(target_length) if target_length else 10,
                target_tone=target_tone,
                target_audience=target_audience,
                creator_style=creator_style,
                content_type=content_type
            )
            
            # Display analysis results
            self.display_script_analysis(validation_result, metadata)
            
        except Exception as e:
            st.error(f"Error analyzing script: {e}")
            st.info("Make sure the script_validator module is properly installed")
    
    def display_script_analysis(self, validation_result, metadata: Dict):
        """Display detailed script analysis results"""
        
        st.markdown("---")
        st.header("📊 Script Analysis Results")
        
        # Overall score
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score_color = "green" if validation_result.overall_score >= 0.7 else "orange" if validation_result.overall_score >= 0.5 else "red"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                <h3 style="color: {score_color}; margin: 0;">Overall Score</h3>
                <h2 style="color: {score_color}; margin: 0;">{validation_result.overall_score:.2f}/1.0</h2>
                <p style="margin: 5px 0 0 0;">{'Excellent' if validation_result.overall_score >= 0.8 else 'Good' if validation_result.overall_score >= 0.6 else 'Needs Improvement'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #e7f3ff; border-radius: 8px;">
                <h3 style="color: #007bff; margin: 0;">Authenticity</h3>
                <h2 style="color: #007bff; margin: 0;">{validation_result.authenticity_score:.2f}/1.0</h2>
                <p style="margin: 5px 0 0 0;">Hinglish Quality</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #fff3cd; border-radius: 8px;">
                <h3 style="color: #856404; margin: 0;">Readability</h3>
                <h2 style="color: #856404; margin: 0;">{validation_result.readability_score:.1f}/100</h2>
                <p style="margin: 5px 0 0 0;">Ease of Reading</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            status_color = "green" if validation_result.is_valid else "red"
            status_text = "PASS" if validation_result.is_valid else "ISSUES"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                <h3 style="color: {status_color}; margin: 0;">Status</h3>
                <h2 style="color: {status_color}; margin: 0;">{status_text}</h2>
                <p style="margin: 5px 0 0 0;">Validation</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed metrics
        st.subheader("📈 Detailed Metrics")
        
        metrics = validation_result.metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Content Metrics:**")
            st.write(f"• Word Count: {metrics['word_count']:,}")
            st.write(f"• Characters: {metrics['char_count']:,}")
            st.write(f"• Est. Duration: {metrics['estimated_minutes']:.1f} min")
            st.write(f"• Target Duration: {metrics['target_minutes']} min")
        
        with col2:
            st.markdown("**Language Analysis:**")
            lang_mix = metrics['language_mix']
            st.write(f"• Hindi: {lang_mix['hindi_ratio']:.1%}")
            st.write(f"• English: {lang_mix['english_ratio']:.1%}")
            st.write(f"• Mixed: {lang_mix['mixed_ratio']:.1%}")
            st.write(f"• Structure Score: {metrics['structure_score']:.2f}/1.0")
        
        with col3:
            st.markdown("**Quality Scores:**")
            st.write(f"• Readability: {metrics['readability_score']:.1f}/100")
            st.write(f"• Engagement: {metrics['engagement_score']}/10")
            st.write(f"• Length Deviation: {metrics['length_deviation']:.1%}")
        
        # Issues and warnings
        if validation_result.issues:
            st.subheader("🚨 Critical Issues")
            for issue in validation_result.issues:
                st.error(f"• {issue}")
        
        if validation_result.warnings:
            st.subheader("⚠️ Warnings")
            for warning in validation_result.warnings:
                st.warning(f"• {warning}")
        
        # Suggestions
        if validation_result.suggestions:
            st.subheader("💡 Improvement Suggestions")
            for suggestion in validation_result.suggestions:
                st.info(f"• {suggestion}")
        
        # Generate and display quality report
        if st.button("📋 Generate Quality Report", key="generate_report"):
            from script_validator import ScriptValidator
            validator = ScriptValidator()
            report = validator.generate_quality_report(validation_result, metadata)
            
            st.subheader("📄 Quality Report")
            st.text_area("Full Quality Report", value=report, height=300, disabled=True)
            
            # Download button for report
            st.download_button(
                label="💾 Download Report",
                data=report,
                file_name=f"script_quality_report_{int(time.time())}.txt",
                mime="text/plain"
            )
    
    def render_personalized_training_tab(self):
        """Render personalized training interface"""
        
        st.header("👤 Personalized Creator Training")
        st.markdown("""
        Upload your own script transcripts to train the system exclusively on your content style.
        When enabled, only your uploaded scripts will be used for training.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Upload Scripts")
            
            creator_id = st.text_input(
                "Creator ID",
                value=st.session_state.creator_id,
                placeholder="Enter your unique creator ID",
                help="A unique identifier for your creator profile (e.g., 'john_doe', 'tech_channel_2024')"
            )
            
            if creator_id:
                st.session_state.creator_id = creator_id
                
                # File uploader
                uploaded_files = st.file_uploader(
                    "Upload Transcript JSON Files",
                    type=['json'],
                    accept_multiple_files=True,
                    help="Upload transcript files in the same JSON format as the existing transcripts"
                )
                
                if uploaded_files and st.button("💾 Save Uploaded Scripts"):
                    from config import Config
                    import json
                    import shutil
                    
                    # Create creator directory
                    creator_dir = Path(Config.USER_SCRIPTS_DIR) / creator_id
                    creator_dir.mkdir(parents=True, exist_ok=True)
                    
                    saved_count = 0
                    for uploaded_file in uploaded_files:
                        try:
                            # Read and validate JSON
                            content = json.load(uploaded_file)
                            
                            # Save to creator directory
                            filename = uploaded_file.name
                            filepath = creator_dir / filename
                            
                            with open(filepath, 'w', encoding='utf-8') as f:
                                json.dump(content, f, ensure_ascii=False, indent=2)
                            
                            saved_count += 1
                        except Exception as e:
                            st.error(f"Error saving {uploaded_file.name}: {e}")
                    
                    if saved_count > 0:
                        st.success(f"✓ Successfully saved {saved_count} script file(s) for creator: {creator_id}")
                        st.info(f"Files saved to: {creator_dir}")
        
        with col2:
            st.subheader("📊 Your Scripts")
            
            if creator_id:
                from config import Config
                from transcript_processor import TranscriptProcessor
                
                creator_dir = Path(Config.USER_SCRIPTS_DIR) / creator_id
                
                if creator_dir.exists():
                    transcript_files = list(creator_dir.glob("*_transcript.json"))
                    
                    if transcript_files:
                        st.metric("Scripts Uploaded", len(transcript_files))
                        
                        st.markdown("**Uploaded Files:**")
                        for file_path in transcript_files:
                            st.markdown(f"• `{file_path.name}`")
                        
                        # Show preview of one transcript
                        if st.button("🔍 Preview Script"):
                            try:
                                processor = TranscriptProcessor()
                                preview_transcript = processor.load_single_transcript(transcript_files[0])
                                if preview_transcript:
                                    st.json({
                                        'title': preview_transcript.metadata.title,
                                        'uploader': preview_transcript.metadata.uploader,
                                        'duration': f"{preview_transcript.metadata.duration // 60} min",
                                        'word_count': preview_transcript.metadata.word_count
                                    })
                            except Exception as e:
                                st.error(f"Error previewing: {e}")
                        
                        # Load and train button
                        if st.button("🔄 Reload & Retrain with Your Scripts"):
                            with st.spinner("Loading and training with your scripts..."):
                                try:
                                    processor = TranscriptProcessor(content_format="short-form")
                                    processed_transcripts = processor.load_user_scripts(creator_id)
                                    
                                    if processed_transcripts:
                                        st.session_state.processed_transcripts = processed_transcripts
                                        st.session_state.creator_summaries = processor.get_creator_summary()
                                        st.session_state.available_creators = processor.get_creator_summary()
                                        
                                        # Train generator
                                        generator = ScriptGenerator()
                                        training_summary = generator.train_on_transcripts(processed_transcripts)
                                        st.session_state.script_generator = generator
                                        st.session_state.training_summary = training_summary
                                        st.session_state.transcripts_loaded = True
                                        st.session_state.script_generator_trained = True
                                        
                                        st.success("✓ Successfully trained with your personalized scripts!")
                                        st.info("You can now use 'Use Personalized Training' in the Generate Script tab.")
                                        st.rerun()
                                    else:
                                        st.error("No valid transcripts found in uploaded files")
                                except Exception as e:
                                    st.error(f"Error training: {e}")
                    else:
                        st.info("No scripts uploaded yet. Upload JSON transcript files using the uploader on the left.")
                else:
                    st.info("Creator directory doesn't exist yet. Upload files to create it.")
            else:
                st.info("Enter a Creator ID to see your uploaded scripts")
        
        # Instructions
        with st.expander("ℹ️ How to Use Personalized Training"):
            st.markdown("""
            **Step 1:** Enter your unique Creator ID
            
            **Step 2:** Upload your transcript JSON files (same format as the training data)
            
            **Step 3:** Click "Save Uploaded Scripts" to store them
            
            **Step 4:** Click "Reload & Retrain with Your Scripts" to train the model on your content
            
            **Step 5:** Go to "Generate Script" tab and enable "Use Personalized Training"
            
            **Important:** 
            - Only your uploaded scripts will be used when personalized training is enabled
            - Make sure your transcript files follow the same JSON structure as the training data
            - The system will learn your unique style, tone, and language patterns
            """)

        # Paste-to-JSON helper
        st.markdown("---")
        st.subheader("📝 Add Script via Text (Convert to JSON)")
        pasted_creator = st.text_input("Creator ID (for saving)", value=st.session_state.get('creator_id', ''), placeholder="your_creator_id")
        pasted_title = st.text_input("Script Title", placeholder="e.g., My Tech Review Script")
        pasted_text = st.text_area("Paste your script text here", height=220, placeholder="Paste plain text script...")
        if st.button("💾 Save Pasted Script as JSON"):
            if not pasted_creator.strip():
                st.error("Please enter a Creator ID.")
            elif not pasted_title.strip():
                st.error("Please enter a Script Title.")
            elif not pasted_text.strip():
                st.error("Please paste the script text.")
            else:
                try:
                    import json, time, re
                    from config import Config
                    # Build segments (>=10). Split by sentences or lines.
                    raw = pasted_text.strip()
                    # Sentence split fallback
                    parts = [p.strip() for p in re.split(r'[.!?\n]+', raw) if p.strip()]
                    if len(parts) < 10:
                        # chunk into ~10 parts
                        words = raw.split()
                        chunks = 10
                        chunk_size = max(1, len(words)//chunks)
                        parts = [" ".join(words[i:i+chunk_size]).strip() for i in range(0, len(words), chunk_size)]
                        parts = [p for p in parts if p]
                        if len(parts) < 10:
                            # pad small lines by splitting further
                            extra = raw
                            while len(parts) < 10:
                                parts.append(extra[:max(1, len(extra)//10)])
                    # Create segments with dummy timing
                    segments = []
                    sec_per = 2
                    t = 0
                    for p in parts:
                        segments.append({"text": p, "start": t, "duration": sec_per})
                        t += sec_per
                    total_words = sum(len(p.split()) for p in parts)
                    from config import Config as Cfg
                    duration_sec = int((total_words / max(1, Cfg.SPEECH_WPM)) * 60)
                    data = {
                        "video_id": f"user_{int(time.time())}",
                        "metadata": {
                            "title": pasted_title.strip(),
                            "uploader": pasted_creator.strip(),
                            "duration": duration_sec,
                            "view_count": 0
                        },
                        "transcript": segments
                    }
                    # Save file
                    creator_dir = Path(Config.USER_SCRIPTS_DIR) / pasted_creator.strip()
                    creator_dir.mkdir(parents=True, exist_ok=True)
                    slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", pasted_title.strip()).strip("-")[:60]
                    filename = f"{slug or 'script'}_transcript.json"
                    out_path = creator_dir / filename
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    st.success(f"✓ Saved to {out_path}")
                    st.info("You can now train using 'Reload & Retrain with Your Scripts'.")
                except Exception as e:
                    st.error(f"Error saving pasted script: {e}")
    
    def export_training_data(self):
        """Export training data for backup"""
        
        if 'script_generator' in st.session_state:
            generator = st.session_state.script_generator
            generator.save_training_context("output/training_context.json")
            st.success("✓ Training context exported to output/training_context.json")
        else:
            st.warning("No training data available to export")

    # ---------------- Thumbnail Tab ----------------
    def _init_gemini_model(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        return genai.GenerativeModel('gemini-2.0-flash')

    def _generate_seo_from_script(self, model, script_text: str, topic: str) -> Dict[str, str]:
        prompt = f"""
You are an expert Indian YouTube strategist. Write in natural Hinglish (Hindi + English mix) suitable for Indian audiences.

Input script (summarized for SEO):
---
{script_text[:2000]}
---

Tasks (use Hinglish, Indian-friendly):
1) Write ONE SEO-friendly YouTube title (<= 70 chars) for this video topic: {topic}.
   - Natural Hinglish, catchy, avoid clickbait cliches
   - No ALL CAPS, no emojis
2) Write an Instagram caption (2–3 short lines) in Hinglish with 3–5 relevant Indian-friendly hashtags.
   - Conversational Hinglish (Latin script), short lines, no overuse of emojis

Format strictly as:
TITLE: <title line>
CAPTION: <caption text on one or multiple lines>
"""
        resp = model.generate_content(prompt, generation_config={"temperature": 0.7, "top_p": 0.9})
        text = resp.text if getattr(resp, 'text', None) else ""
        title, caption = "", ""
        for line in (text or "").split('\n'):
            if line.upper().startswith("TITLE:") and not title:
                title = line.split(":", 1)[-1].strip()
            elif line.upper().startswith("CAPTION:") and not caption:
                caption = line.split(":", 1)[-1].strip()
            elif caption:
                caption += ("\n" + line.strip())
        return {"title": title[:70], "caption": caption.strip()}

    def render_thumbnail_studio(self):
        """Render thumbnail generator with split-screen layout"""
        
        st.header("🖼️ Thumbnail Studio")
        st.markdown("Generate cinematic thumbnails with SEO titles and Instagram captions")
        
        # Split screen layout: 30% controls, 70% preview
        left_col, right_col = st.columns([3, 7])
        
        with left_col:
            st.subheader("⚙️ Controls")
            
            # Script input
            script_text = st.text_area(
                "Paste Script",
                height=150,
                placeholder="Paste your generated script here...",
                help="The script will be analyzed to generate SEO content"
            )
            
            # Basic settings
            topic = st.text_input(
                "Topic",
                placeholder="e.g., iPhone 15 Pro Review",
                help="Main topic for SEO and thumbnail prompt"
            )
            
            tone = st.selectbox(
                "Mood/Tone",
                options=["dramatic", "inspiring", "energetic", "mysterious", "friendly"],
                index=0,
                help="Overall mood for the thumbnail"
            )
            
            # Image settings
            with st.expander("🖼️ Image Settings"):
                model_choice = st.selectbox(
                    "Model",
                    ["FLUX.1-schnell (Fast, Good Quality)", "SDXL-Turbo (Very Fast)", "SDXL-Base (Best Quality, Slower)"],
                    index=0,
                    help="FLUX.1-schnell recommended for best balance"
                )
                width = st.number_input("Width", min_value=640, max_value=2048, value=1280, step=64)
                height = st.number_input("Height", min_value=360, max_value=2048, value=720, step=40)
                steps = st.slider("Quality Steps", min_value=4, max_value=30, value=8, help="Higher = better quality but slower. 8-12 recommended for FLUX")
                guidance = st.slider("Guidance Scale", min_value=0.0, max_value=10.0, value=3.5, step=0.5, help="How closely to follow prompt. 3-5 recommended")
                enhance_quality = st.checkbox("Apply Quality Enhancement", value=True, help="Post-process for YouTube optimization")
            
            # Overlay settings
            with st.expander("📝 Text Overlay Settings"):
                add_overlay = st.checkbox("Add Title Overlay", value=True)
                position = st.selectbox("Position", ["auto", "left", "right", "topbar"], index=0)
                stroke = st.slider("Stroke Width", 0, 12, 6)
                custom_text = st.text_input("Custom Overlay Text", value="", help="Leave empty to use SEO title")
                font_file = st.file_uploader("Upload Font (.ttf)", type=["ttf"], help="Optional: Upload a custom font")
            
            # Generate button
            if st.button("🎨 Generate Thumbnail + SEO", type="primary", use_container_width=True):
                if not script_text.strip():
                    st.toast("❌ Please paste the script", icon="❌")
                else:
                    # Store for right column display
                    st.session_state.thumbnail_generating = True
                    st.session_state.thumbnail_script = script_text
                    st.session_state.thumbnail_topic = topic
                    st.session_state.thumbnail_tone = tone
                    st.session_state.thumbnail_width = width
                    st.session_state.thumbnail_height = height
                    st.session_state.thumbnail_steps = steps
                    st.session_state.thumbnail_guidance = guidance
                    st.session_state.thumbnail_model = model_choice
                    st.session_state.thumbnail_enhance = enhance_quality
                    st.session_state.thumbnail_add_overlay = add_overlay
                    st.session_state.thumbnail_position = position
                    st.session_state.thumbnail_stroke = stroke
                    st.session_state.thumbnail_custom_text = custom_text
                    st.session_state.thumbnail_font_file = font_file
                    st.rerun()
        
        with right_col:
            st.subheader("👁️ Preview")
            
            # Generate thumbnail if button was clicked
            if st.session_state.get('thumbnail_generating', False):
                script_text = st.session_state.thumbnail_script
                topic = st.session_state.thumbnail_topic
                tone = st.session_state.thumbnail_tone
                width = st.session_state.thumbnail_width
                height = st.session_state.thumbnail_height
                steps = st.session_state.thumbnail_steps
                guidance = st.session_state.thumbnail_guidance
                model_choice = st.session_state.thumbnail_model
                enhance_quality = st.session_state.thumbnail_enhance
                add_overlay = st.session_state.thumbnail_add_overlay
                position = st.session_state.thumbnail_position
                stroke = st.session_state.thumbnail_stroke
                custom_text = st.session_state.thumbnail_custom_text
                font_file = st.session_state.thumbnail_font_file
                
                # Determine model ID based on choice
                model_id_map = {
                    "FLUX.1-schnell (Fast, Good Quality)": "black-forest-labs/FLUX.1-schnell",
                    "SDXL-Turbo (Very Fast)": "stabilityai/sdxl-turbo",
                    "SDXL-Base (Best Quality, Slower)": "stabilityai/stable-diffusion-xl-base-1.0"
                }
                selected_model = model_id_map.get(model_choice, "black-forest-labs/FLUX.1-schnell")
                
                # Generate thumbnail with better prompt
                built = build_thumbnail_prompt(topic or "YouTube video", script_text, tone)
                with st.spinner("🎨 Generating high-quality thumbnail... This may take 30-60 seconds"):
                    img_bytes = generate_thumbnail(
                        prompt=built["prompt"],
                        negative_prompt=built["negative"],
                        width=int(width),
                        height=int(height),
                        steps=int(steps),
                        guidance=float(guidance),
                        model_id=selected_model,
                    )
                
                if not img_bytes:
                    st.error("❌ Thumbnail generation failed. Check API token or retry.")
                    st.toast("❌ Generation failed", icon="❌")
                else:
                    # Apply quality enhancement if enabled
                    if enhance_quality:
                        try:
                            from thumbnail_overlay import apply_youtube_optimization
                            img_bytes = apply_youtube_optimization(img_bytes)
                            st.toast("✓ Quality enhancement applied!", icon="✅")
                        except Exception as e:
                            st.warning(f"Quality enhancement failed: {e}")
                    
                    # Generate SEO content
                    seo = {}
                    try:
                        model = self._init_gemini_model()
                        seo = self._generate_seo_from_script(model, script_text, topic or "")
                        st.toast("✓ SEO content generated!", icon="✅")
                    except Exception as e:
                        st.warning(f"SEO generation failed: {e}")
                    
                    # Apply overlay
                    final_image = img_bytes
                    if add_overlay and img_bytes:
                        title_text = custom_text.strip() or seo.get("title", "") or (topic or "")
                        if title_text:
                            try:
                                font_path = None
                                # Handle font file if uploaded
                                if font_file is not None:
                                    # Save to temp file for PIL
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
                                        tmp_file.write(font_file.read())
                                        font_path = tmp_file.name
                                
                                final_image = render_text(
                                    image_bytes=img_bytes,
                                    text=title_text,
                                    position=position,
                                    font_path=font_path,
                                    text_color=(255, 255, 255),
                                    stroke_color=(0, 0, 0),
                                    stroke_width=int(stroke),
                                    shadow_color=(0, 0, 0),
                                    shadow_offset=4,
                                )
                            except Exception as e:
                                st.warning(f"Overlay failed: {e}")
                    
                    # Display thumbnail
                    st.image(final_image, caption="Generated Thumbnail", width=None)
                    
                    # Download button
                    slug = (topic or "thumbnail").strip().replace(" ", "-")[:50]
                    st.download_button(
                        "📥 Download Thumbnail",
                        data=final_image,
                        file_name=f"{slug}_thumbnail.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    # SEO Content Display
                    st.markdown("---")
                    st.subheader("📝 SEO Content")
                    
                    # SEO Title
                    if seo.get("title"):
                        st.markdown("**SEO Title:**")
                        st.code(seo["title"], language=None)
                        col1, col2 = st.columns(2)
                        with col1:
                            self._copy_to_clipboard_button(seo["title"], "Copy Title")
                        with col2:
                            st.markdown(f"*{len(seo['title'])} characters*")
                    
                    # Instagram Caption
                    if seo.get("caption"):
                        st.markdown("**Instagram Caption:**")
                        st.code(seo["caption"], language=None)
                        col1, col2 = st.columns(2)
                        with col1:
                            self._copy_to_clipboard_button(seo["caption"], "Copy Caption")
                        with col2:
                            st.markdown(f"*{len(seo['caption'])} characters*")
                    
                    # Reset flag
                    st.session_state.thumbnail_generating = False
            else:
                st.info("👈 Configure settings and click 'Generate Thumbnail + SEO' to create your thumbnail")
    
    def render_content_planner(self):
        """Render content planning and calendar interface"""
        st.header("📅 Content Planner")
        st.markdown("Plan your content strategy with trend analysis and calendar")
        
        # Initialize calendar
        try:
            from content_calendar import ContentCalendar
            calendar = ContentCalendar()
        except Exception as e:
            st.error(f"Failed to load content calendar: {e}")
            return
        
        # Tabs for different planning features
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Trends", "📅 Calendar", "💡 Topic Suggestions", "🔄 Batch Generate"])
        
        with tab1:
            self._render_trends_tab()
        
        with tab2:
            self._render_calendar_tab(calendar)
        
        with tab3:
            self._render_topic_suggestions_tab()
        
        with tab4:
            self._render_batch_generate_tab()
    
    def _render_trends_tab(self):
        """Render trends analysis tab"""
        st.subheader("📈 Trend Analysis")
        
        niche = st.text_input("Enter your niche", placeholder="e.g., Technology, Entertainment, Education")
        
        if st.button("🔍 Analyze Trends", type="primary"):
            if not niche:
                st.toast("❌ Please enter a niche", icon="❌")
                return
            
            try:
                from trends_analyzer import TrendsAnalyzer
                analyzer = TrendsAnalyzer()
                
                with st.spinner("Analyzing trends..."):
                    trends = analyzer.analyze_niche_trends(niche)
                
                if trends.get('trending_videos'):
                    st.subheader("🔥 Trending Videos")
                    for video in trends['trending_videos'][:10]:
                        with st.expander(f"📺 {video.get('title', 'Unknown')}"):
                            st.write(f"**Channel:** {video.get('channel', 'Unknown')}")
                            st.write(f"**Published:** {video.get('published_at', 'Unknown')}")
                            st.write(f"**Description:** {video.get('description', 'No description')}")
                
                if trends.get('popular_keywords'):
                    st.subheader("🔑 Popular Keywords")
                    st.write(", ".join(trends['popular_keywords'][:15]))
                
                if trends.get('insights'):
                    st.subheader("💡 Insights")
                    for insight in trends['insights']:
                        st.info(insight)
                
            except Exception as e:
                st.error(f"Trend analysis failed: {e}")
                st.toast(f"❌ Error: {e}", icon="❌")
    
    def _render_calendar_tab(self, calendar):
        """Render content calendar tab"""
        st.subheader("📅 Content Calendar")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Add New Video Idea")
            topic = st.text_input("Topic", placeholder="e.g., iPhone 16 Review")
            scheduled_date = st.date_input("Scheduled Date")
            niche = st.text_input("Niche (optional)", placeholder="e.g., Technology")
            notes = st.text_area("Notes (optional)", height=100)
            
            if st.button("➕ Add to Calendar", type="primary"):
                if topic:
                    idea = calendar.add_idea(topic, scheduled_date.isoformat(), niche, notes)
                    st.toast(f"✓ Added: {topic}", icon="✅")
                    st.rerun()
                else:
                    st.toast("❌ Please enter a topic", icon="❌")
        
        with col2:
            st.markdown("### Calendar Stats")
            stats = calendar.get_statistics()
            st.metric("Total Ideas", stats['total_ideas'])
            st.metric("Upcoming (30d)", stats['upcoming_30_days'])
            st.metric("Script Ready", stats['script_ready'])
            st.metric("Thumbnail Ready", stats['thumbnail_ready'])
        
        # Show upcoming ideas
        st.markdown("### 📋 Upcoming Ideas")
        upcoming = calendar.get_upcoming_ideas(30)
        
        if upcoming:
            for idea in upcoming:
                with st.expander(f"📌 {idea.topic} - {idea.scheduled_date}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Status:** {idea.status}")
                        st.write(f"**Niche:** {idea.niche or 'Not specified'}")
                    with col2:
                        if idea.script_generated:
                            st.success("✓ Script Ready")
                        if idea.thumbnail_generated:
                            st.success("✓ Thumbnail Ready")
                        if idea.seo_ready:
                            st.success("✓ SEO Ready")
                    
                    if idea.notes:
                        st.write(f"**Notes:** {idea.notes}")
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{idea.id}"):
                        calendar.delete_idea(idea.id)
                        st.toast("✓ Idea deleted", icon="✅")
                        st.rerun()
        else:
            st.info("No upcoming ideas. Add some to get started!")
    
    def _render_topic_suggestions_tab(self):
        """Render topic suggestions tab"""
        st.subheader("💡 AI Topic Suggestions")
        
        niche = st.text_input("Your Niche", placeholder="e.g., Technology Reviews")
        count = st.slider("Number of suggestions", 5, 20, 10)
        
        if st.button("🎯 Generate Topic Ideas", type="primary"):
            if not niche:
                st.toast("❌ Please enter a niche", icon="❌")
                return
            
            try:
                from ai_agents import ScriptResearchAgent
                agent = ScriptResearchAgent()
                
                with st.spinner("Generating topic ideas..."):
                    topics = agent.suggest_topics(niche, count)
                
                if topics:
                    st.subheader("✨ Suggested Topics")
                    for idx, topic_data in enumerate(topics, 1):
                        with st.expander(f"{idx}. {topic_data.get('topic', 'Unknown')}"):
                            st.write(f"**Why:** {topic_data.get('reason', 'N/A')}")
                            st.write(f"**Audience:** {topic_data.get('target_audience', 'N/A')}")
                            st.write(f"**Engagement:** {topic_data.get('engagement', 'N/A')}")
                            
                            if st.button(f"📅 Add to Calendar", key=f"add_topic_{idx}"):
                                try:
                                    from content_calendar import ContentCalendar
                                    calendar = ContentCalendar()
                                    calendar.add_idea(
                                        topic_data.get('topic', ''),
                                        datetime.now().date().isoformat(),
                                        niche
                                    )
                                    st.toast("✓ Added to calendar!", icon="✅")
                                except Exception as e:
                                    st.error(f"Failed to add: {e}")
                else:
                    st.warning("No topics generated. Try a different niche.")
                    
            except Exception as e:
                st.error(f"Topic generation failed: {e}")
                st.toast(f"❌ Error: {e}", icon="❌")
    
    def _render_batch_generate_tab(self):
        """Render batch generation tab"""
        st.subheader("🔄 Batch Script Generation")
        st.markdown("Generate multiple scripts at once for content planning")
        
        if not st.session_state.transcripts_loaded or not st.session_state.script_generator_trained:
            st.warning("⚠️ Please initialize the system from Dashboard first")
            return
        
        # Input method
        input_method = st.radio("Input Method", ["Manual Entry", "From Calendar"], horizontal=True)
        
        topics = []
        
        if input_method == "Manual Entry":
            topics_text = st.text_area(
                "Enter topics (one per line)",
                height=200,
                placeholder="iPhone 16 Review\nSamsung Galaxy S24 Unboxing\nTop 5 Budget Phones"
            )
            if topics_text:
                topics = [t.strip() for t in topics_text.split('\n') if t.strip()]
        else:
            try:
                from content_calendar import ContentCalendar
                calendar = ContentCalendar()
                upcoming = calendar.get_upcoming_ideas(90)
                
                if upcoming:
                    selected_ideas = st.multiselect(
                        "Select ideas to generate scripts for",
                        options=[idea.topic for idea in upcoming],
                        default=[idea.topic for idea in upcoming[:5]]
                    )
                    topics = selected_ideas
                else:
                    st.info("No ideas in calendar. Add some ideas first!")
            except Exception as e:
                st.error(f"Failed to load calendar: {e}")
        
        if topics:
            st.info(f"📝 Will generate {len(topics)} scripts")
            
            # Base parameters
            with st.expander("⚙️ Base Parameters (applied to all)"):
                base_tone = st.selectbox("Tone", Config.VALID_TONES, index=0)
                base_audience = st.selectbox("Target Audience", Config.VALID_AUDIENCES, index=0)
                base_content_type = st.selectbox("Content Type", ['review', 'comparison', 'unboxing', 'tutorial', 'general'], index=0)
                base_length = st.selectbox("Length (minutes)", [5, 8, 10, 12, 15, 18, 20], index=2)
                base_language_mix = st.select_slider("Language Mix", ['Hindi Heavy', 'Balanced', 'English'], value='Balanced')
            
            if st.button("🚀 Generate Batch Scripts", type="primary", use_container_width=True):
                try:
                    from batch_processor import BatchProcessor
                    processor = BatchProcessor(st.session_state.script_generator)
                    
                    base_params = {
                        'length_minutes': base_length,
                        'tone': base_tone,
                        'target_audience': base_audience,
                        'content_type': base_content_type,
                        'language_mix': base_language_mix,
                        'content_format': 'long-form'
                    }
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(completed, total, current_topic):
                        progress_bar.progress(completed / total)
                        status_text.text(f"Generating {completed}/{total}: {current_topic}")
                    
                    results = processor.generate_batch_scripts(topics, base_params, progress_callback)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✓ Batch generation complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Generation Results")
                    
                    success_count = sum(1 for r in results if r.get('success'))
                    st.metric("Success Rate", f"{success_count}/{len(results)}")
                    
                    # Show results
                    for result in results:
                        if result.get('success'):
                            with st.expander(f"✓ {result.get('batch_topic', 'Unknown')}"):
                                st.write(f"**Status:** Success")
                                if 'script' in result:
                                    st.code(result['script'][:500] + "...", language=None)
                        else:
                            with st.expander(f"❌ {result.get('batch_topic', 'Unknown')}"):
                                st.error(f"**Error:** {result.get('error', 'Unknown error')}")
                    
                    # Export results
                    if st.button("📥 Export Results as CSV"):
                        try:
                            from export_tools import ExportTools
                            csv_file = ExportTools.export_batch_csv(results)
                            st.toast(f"✓ Exported to {csv_file}", icon="✅")
                        except Exception as e:
                            st.error(f"Export failed: {e}")
                    
                except Exception as e:
                    st.error(f"Batch generation failed: {e}")
                    st.toast(f"❌ Error: {e}", icon="❌")
    
    def render_analytics_dashboard(self):
        """Render analytics dashboard"""
        st.header("📊 Analytics Dashboard")
        st.markdown("Track your script generation performance and usage patterns")
        
        try:
            from analytics import Analytics
            analytics = Analytics()
        except Exception as e:
            st.error(f"Failed to load analytics: {e}")
            return
        
        stats = analytics.get_statistics()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Scripts", stats['total_scripts'])
        
        with col2:
            st.metric("Total Thumbnails", stats['total_thumbnails'])
        
        with col3:
            st.metric("Weekly Scripts", stats['weekly_scripts'])
        
        with col4:
            st.metric("Success Rate", f"{stats['success_rate']}%")
        
        # Charts and breakdowns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Top Tones")
            if stats['top_tones']:
                for tone, count in stats['top_tones']:
                    st.write(f"**{tone.replace('_', ' ').title()}:** {count}")
            else:
                st.info("No data yet")
        
        with col2:
            st.subheader("👥 Top Audiences")
            if stats['top_audiences']:
                for audience, count in stats['top_audiences']:
                    st.write(f"**{audience.replace('_', ' ').title()}:** {count}")
            else:
                st.info("No data yet")
        
        # Daily trends
        st.subheader("📅 Daily Trends (Last 30 Days)")
        trends = analytics.get_daily_trends(30)
        
        if trends:
            import pandas as pd
            df = pd.DataFrame(trends)
            st.line_chart(df.set_index('date')[['scripts', 'thumbnails']])
        
        # Recent activity
        st.subheader("🕐 Recent Activity")
        try:
            recent = analytics.data.get('recent_activity', [])[:10]
            if recent:
                for activity in recent:
                    status_icon = "✅" if activity.get('success') else "❌"
                    st.write(f"{status_icon} **{activity.get('type', 'unknown')}** - {activity.get('topic', 'Unknown')} ({activity.get('timestamp', '')[:10]})")
            else:
                st.info("No recent activity")
        except Exception as e:
            st.warning(f"Could not load recent activity: {e}")

# Main app execution
def main():
    """Main function to run the Streamlit app"""
    
    try:
        app = YouTubeScriptGeneratorApp()
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please check your configuration and try again.")

if __name__ == "__main__":
    main()

