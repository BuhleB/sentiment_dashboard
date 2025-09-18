# Sentiment Analysis Dashboard

An interactive web application for analyzing sentiment in text data with comprehensive visualization and export capabilities.

## Features

### Core Sentiment Analysis
- **Multi-class Classification**: Automatically classifies text as Positive, Negative, or Neutral
- **Confidence Scoring**: Provides confidence scores for each classification
- **Keyword Extraction**: Identifies key words that drive sentiment
- **Batch Processing**: Analyze multiple texts simultaneously
- **Explanation Features**: Detailed explanations for why text received specific sentiment scores

### Interactive Dashboard
- **Real-time Analysis**: Instant sentiment analysis as you type
- **Multiple Input Methods**: 
  - Single text input
  - Batch text analysis
  - File upload (CSV and TXT files)
- **Advanced Filtering**: Filter results by sentiment, source, and other criteria

### Comprehensive Visualizations
- **Sentiment Distribution**: Pie charts showing overall sentiment breakdown
- **Confidence Analysis**: Histograms of confidence score distributions
- **Keyword Analysis**: Word clouds and frequency charts
- **Polarity vs Subjectivity**: Scatter plots for detailed analysis
- **Trend Analysis**: Time-based sentiment trends
- **Source Comparison**: Compare sentiment across different data sources

### Export Capabilities
- **Multiple Formats**: Export results in CSV, JSON, Excel, and PDF formats
- **Comprehensive Reports**: PDF reports with executive summaries and detailed analysis
- **Data Preservation**: All analysis results can be saved for future reference

## Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK data (optional, fallback included):
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

## Usage

### Running the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Input Methods

#### Single Text Analysis
1. Select "Single Text Analysis" from the sidebar
2. Enter your text in the text area
3. Optionally specify source and date
4. Click "Analyze Text"

#### Batch Text Analysis
1. Select "Batch Text Analysis" from the sidebar
2. Enter multiple texts (one per line)
3. Specify default source and date
4. Click "Analyze Batch"

#### File Upload Analysis
1. Select "File Upload Analysis" from the sidebar
2. Upload CSV or TXT files
3. For CSV files, ensure there's a 'text' column
4. Click "Analyze All Files" or "Analyze Single File"

### Viewing Results

The dashboard provides multiple tabs for viewing results:

- **Overview**: Sentiment distribution, confidence analysis, and polarity vs subjectivity plots
- **Trends**: Time-based sentiment analysis (when date information is available)
- **Keywords**: Word clouds and keyword frequency analysis
- **Data**: Detailed data table with export options

### Exporting Results

1. Navigate to the "Data" tab
2. Choose your preferred export format (CSV, JSON, Excel, or PDF Report)
3. Click "Download Data"

## Technical Details

### Sentiment Analysis Engine
- **Primary Library**: TextBlob for sentiment analysis
- **Keyword Extraction**: NLTK with stop-word filtering
- **Fallback Support**: Built-in fallbacks when NLTK data is unavailable

### Visualization Libraries
- **Plotly**: Interactive charts and graphs
- **Matplotlib**: Word cloud generation
- **Streamlit**: Web interface and dashboard

### Export Libraries
- **Pandas**: Data manipulation and CSV/Excel export
- **ReportLab**: PDF report generation
- **OpenPyXL**: Excel file handling

## Sample Data

The project includes `sample_data.csv` with example customer reviews and social media posts for testing the application.

## File Structure

```
sentiment_dashboard/
├── app.py                 # Main Streamlit application
├── sentiment_analyzer.py  # Core sentiment analysis functions
├── visualizations.py      # Chart and graph generation
├── export_utils.py        # Export functionality
├── requirements.txt       # Python dependencies
├── sample_data.csv        # Sample data for testing
├── nltk_data/            # NLTK data directory
└── README.md             # This file
```

## Customization

### Adding New Sentiment Models
Modify `sentiment_analyzer.py` to integrate additional sentiment analysis libraries or models.

### Custom Visualizations
Add new chart types in `visualizations.py` and integrate them into the dashboard tabs.

### Export Formats
Extend `export_utils.py` to support additional export formats or customize existing ones.

## Troubleshooting

### NLTK Data Issues
If you encounter NLTK data errors, the application includes fallback mechanisms. For full functionality, download NLTK data:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Memory Issues with Large Files
For very large datasets, consider processing files in smaller batches or increasing system memory.

### Port Conflicts
If port 8501 is in use, specify a different port:
```bash
streamlit run app.py --server.port 8502
```

## License

This project is open source and available under the MIT License.

