# 🎬 YouTube Script Generator

**Generate authentic Hinglish YouTube scripts that capture the unique style of Indian tech creators**

![WhatsApp Image 2025-11-26 at 22 12 41_80642306](https://github.com/user-attachments/assets/ee5d51af-cc89-4b1f-9738-c5fc51da396c)
![WhatsApp Image 2025-11-26 at 22 15 30_e3a54bc5](https://github.com/user-attachments/assets/ffd60085-ea7e-44e8-ba0b-6190e1c295fb)
![WhatsApp Image 2025-11-26 at 22 19 18_019198d0](https://github.com/user-attachments/assets/16f8be84-8c73-46a4-b0a9-21ed17ceedb7)
![WhatsApp Image 2025-11-26 at 22 21 02_5e6511ad](https://github.com/user-attachments/assets/f67d249e-5417-4a6c-b8f2-75548fb8a0a4)


A comprehensive AI-powered tool that analyzes existing YouTube transcripts from Indian tech creators and generates new scripts that authentically replicate their speaking patterns, language mix, and content style.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red.svg)
![Gemini](https://img.shields.io/badge/Gemini-API-green.svg)

## ✨ Features

### 🧠 Intelligent Analysis

- **Creator Style Extraction**: Analyzes individual creator patterns, tone, and language preferences
- **Hinglish Understanding**: Automatically detects and replicates authentic Hindi-English mixing
- **Content Classification**: Identifies video types, duration patterns, and engagement elements
- **Language Mix Optimization**: Balances Hindi and English based on creator habits

### 🎭 Authentic Style Replication

- **Trakins Tech Style**: Enthusiastic, tech-focused reviews with detailed specifications
- **TechBar Style**: Comprehensive analysis with technical depth
- **Custom Creator Matching**: Automatically selects best matching style or manual override
- **Tone Adaptation**: Friendly, professional, energetic, or technical as needed

### 📝 Advanced Script Generation

- **Length Control**: Precise timing for target duration (2-40 minutes supported)
- **Content Types**: Reviews, comparisons, unboxing, tutorials, news, guides
- **Audience Targeting**: Tailored for tech enthusiasts, beginners, professionals, etc.
- **Structured Output**: Proper YouTube format with hooks, intros, CTAs, outros

### 🔧 Comprehensive Parameters

- **Length**: Target video duration in minutes
- **Tone**: 7 different tone options from enthusiastic to technical
- **Audience**: 7 target audience specifications
- **Content Type**: Multiple YouTube content formats
- **Creator Style**: Auto-match or manual creator selection
- **Language Mix**: Hindi-heavy, balanced, or English-heavy preferences

### 🛠️ Quality Validation

- **Authenticity Scoring**: Measures Hinglish authenticity and creator consistency
- **Readability Analysis**: Adapts complexity for target audience
- **Structure Validation**: Ensures proper video flow and engagement elements
- **Length Optimization**: Validates word count against speaking time
- **Improvement Suggestions**: Actionable recommendations for script enhancement

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- YouTube transcript data in JSON format

### Installation

1. **Clone or download the project**

   ```bash
   # If you have git (optional)
   git clone <your-repo-url>
   cd youtube-script-generator
   ```

2. **Set up environment**

   ```bash
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure API key**
   Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Prepare transcript data**

   - Place your JSON transcript files in `Data/processed/`
   - Files should be named `*_transcript.json`
   - Format: Each file contains video metadata and segmented transcript

5. **Run the application**

   ```bash
   python run_app.py
   ```

   Or directly with Streamlit:

   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - Navigate to http://localhost:8501
   - Follow the setup steps in the sidebar

## 📊 Data Format

Your transcript files should follow this structure:

```json
{
  "video_id": "example_video_id",
  "metadata": {
    "title": "Video Title",
    "uploader": "Creator Name",
    "duration": 596,
    "view_count": 1206493
  },
  "transcript": [
    {
      "text": "दोस्तों ये वीडियो हम हर साल करते हैं",
      "start": 0.04,
      "duration": 3.72
    }
  ]
}
```

## 🎯 Usage Guide

### Step 1: Load Transcripts

- Click "Load Transcript Data" in the sidebar
- System analyzes all available transcript files
- Extracts creator styles and patterns

### Step 2: Train Generator

- Click "Train Script Generator"
- AI learns from your transcript data
- Creates style models for each creator

### Step 3: Generate Scripts

1. **Enter Topic**: e.g., "iPhone 15 Pro Max Review"
2. **Set Duration**: Choose target length (5-25 minutes recommended)
3. **Select Content Type**: Review, comparison, tutorial, etc.
4. **Choose Style Parameters**:
   - Creator Style: Auto or specific creator
   - Tone: Friendly, enthusiastic, technical, etc.
   - Target Audience: Tech enthusiasts, beginners, etc.
   - Language Mix: Balanced, Hindi-heavy, English-heavy
5. **Click Generate**: AI creates authentic script

### Step 4: Review & Validate

- System automatically validates generated script
- Check authenticity and readability scores
- Review improvement suggestions
- Save or regenerate variations

## 🔧 Configuration

### Environment Variables

Create a `.env` file with these variables:

```env
# Required
GEMINI_API_KEY=your_api_key_here

# Optional
DEBUG=True
NUM_TRANSCRIPTS_TO_LOAD=25
MAX_SCRIPT_LENGTH_CHARS=20000
DEFAULT_TONE=friendly_and_informative
DEFAULT_TARGET_AUDIENCE=tech_enthusiasts
DEFAULT_LENGTH_MINUTES=10
```

### Settings

- **Valid Tones**: friendly_and_informative, enthusiastic_and_energetic, professional_and_formal, casual_and_conversational, dramatic_and_engaging, technical_and_detailed, humorous_and_entertaining
- **Target Audiences**: tech_enthusiasts, general_audience, beginners, professionals, students, gamers, content_creators
- **Content Types**: review, comparison, unboxing, tutorial, general

## 📈 Features Deep Dive

### 🧠 Style Analysis

- **Language Patterns**: Analyzes Hindi-English mixing ratios
- **Speaking Pace**: Measures words per minute across creators
- **Engagement Patterns**: Identifies CTA styles and audience engagement
- **Technical Focus**: Detects level of technical detail by creator
- **Unique Expressions**: Captures creator-specific phrases and mannerisms

### 🎭 Creator Profiles

- **Trakin Tech**: Enthusiastic smartphone reviews with detailed specifications and pricing
- **TechBar**: Comprehensive technical analysis with longer-form content
- **Auto-Detection**: System automatically selects best matching style for any topic
- **Style Fusion**: Ability to blend multiple creator styles

### ✅ Validation System

- **Authenticity Score**: Measures how well script matches creator style (0-1)
- **Readability Score**: Ensures appropriate complexity for audience (0-100)
- **Structure Validation**: Checks for proper video flow elements
- **Engagement Check**: Validates presence of CTAs and engagement elements
- **Length Optimization**: Ensures word count matches target duration

## 🔍 Troubleshooting

### Common Issues

**API Key Error**

```
Solution: Ensure GEMINI_API_KEY is correctly set in .env file
```

**No Transcripts Found**

```
Solution: Ensure files are in Data/processed/ with *_transcript.json format
```

**Memory Issues**

```
Solution: Reduce NUM_TRANSCRIPTS_TO_LOAD in .env file
```

**Generation Fails**

```
Solution: Check Gemini API quotas and ensure sufficient credits
```

### Performance Optimization

- **Large Datasets**: Increase NUM_TRANSCRIPTS_TO_LOAD gradually
- **Fast Generation**: Use Gemini 1.5 Flash (default) for speed
- **Memory Management**: Restart app periodically for large transcript collections

## 📝 Example Outputs

### Generated Script Sample

```
[HOOK]
दोस्तों, क्या आप नया smartphone खरीदना चाहते हैं लेकिन confuse हैं कि कौन सा लें?

[INTRO]
हैलो tech enthusiasts, मैं आज आपके लिए लाया हूं एक detailed review...

[MAIN CONTENT]
इस phone में आपको मिलने वाले हैं तीन main cameras जिसमें से primary sensor...
The performance with Snapdragon 8 Gen 3 is absolutely fantastic for gaming...

[CTA]
अगर यह review आपको helpful लगी हो तो please like और subscribe करें...

[OUTRO]
तो दोस्तों, यहाँ था हमारा today का review. अगले video में मिलते हैं...
```

## 🔮 Advanced Features

### Content Analysis Tab

- View detailed creator breakdowns
- Analyze language distribution across creators
- Compare tone patterns and style markers
- Export analysis data

### Creator Styles Tab

- Deep dive into individual creator analysis
- View tone profiles and language preferences
- See common keywords and style markers
- Understand creator-specific patterns

### Settings Tab

- Modify generation parameters
- View current configuration
- Export/import training data
- Advanced model settings

## 🤝 Contributing

This tool is designed for creators analyzing existing YouTube content. To contribute:

1. **Add New Creator Styles**: Extend Config.YOUTUBE_CREATORS dictionary
2. **Improve Validation**: Enhance script_validator.py metrics
3. **New Content Types:** Add support for additional video formats
4. **Language Patterns**: Improve Hinglish detection and generation

## 📄 License

This project is for educational and research purposes. Ensure compliance with YouTube's terms of service and copyright policies when using with creator content.

## 🙏 Acknowledgments

- Transcript data from Indian tech creators for style analysis
- Google Gemini API for advanced language generation
- Streamlit for intuitive web interface
- Hindi-English language processing libraries

---

**Ready to create authentic YouTube scripts?** 🚀

Start with `python run_app.py` and generate your first script in minutes!

