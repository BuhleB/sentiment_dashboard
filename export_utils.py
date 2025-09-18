import pandas as pd
import json
import csv
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from visualizations import create_sentiment_metrics_summary

def export_to_csv(df, filename=None):
    """Export dataframe to CSV format"""
    if filename is None:
        filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    csv_data = df.to_csv(index=False)
    return csv_data, filename

def export_to_json(df, filename=None):
    """Export dataframe to JSON format"""
    if filename is None:
        filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    json_data = df.to_json(orient='records', indent=2)
    return json_data, filename

def export_to_excel(df, filename=None):
    """Export dataframe to Excel format"""
    if filename is None:
        filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='Sentiment Analysis', index=False)
        
        # Summary sheet
        metrics = create_sentiment_metrics_summary(df)
        summary_data = {
            'Metric': [
                'Total Texts',
                'Positive Count',
                'Negative Count',
                'Neutral Count',
                'Positive Percentage',
                'Negative Percentage',
                'Neutral Percentage',
                'Average Confidence',
                'Average Polarity',
                'Average Subjectivity'
            ],
            'Value': [
                metrics['total_texts'],
                metrics['positive_count'],
                metrics['negative_count'],
                metrics['neutral_count'],
                f"{metrics['positive_pct']:.1f}%",
                f"{metrics['negative_pct']:.1f}%",
                f"{metrics['neutral_pct']:.1f}%",
                f"{metrics['avg_confidence']:.3f}",
                f"{metrics['avg_polarity']:.3f}",
                f"{metrics['avg_subjectivity']:.3f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    excel_data = output.getvalue()
    return excel_data, filename

def create_pdf_report(df, filename=None):
    """Create a comprehensive PDF report"""
    if filename is None:
        filename = f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Create PDF buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20
    )
    
    # Build story
    story = []
    
    # Title
    story.append(Paragraph("Sentiment Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    metrics = create_sentiment_metrics_summary(df)
    
    summary_text = f"""
    This report analyzes {metrics['total_texts']} text samples for sentiment classification.
    
    Key Findings:
    • {metrics['positive_count']} texts ({metrics['positive_pct']:.1f}%) were classified as Positive
    • {metrics['negative_count']} texts ({metrics['negative_pct']:.1f}%) were classified as Negative  
    • {metrics['neutral_count']} texts ({metrics['neutral_pct']:.1f}%) were classified as Neutral
    
    The average confidence score across all classifications was {metrics['avg_confidence']:.3f}, 
    with an overall polarity score of {metrics['avg_polarity']:.3f} and subjectivity score of {metrics['avg_subjectivity']:.3f}.
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Detailed Metrics Table
    story.append(Paragraph("Detailed Metrics", heading_style))
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Texts Analyzed', str(metrics['total_texts'])],
        ['Positive Sentiment Count', str(metrics['positive_count'])],
        ['Negative Sentiment Count', str(metrics['negative_count'])],
        ['Neutral Sentiment Count', str(metrics['neutral_count'])],
        ['Positive Percentage', f"{metrics['positive_pct']:.1f}%"],
        ['Negative Percentage', f"{metrics['negative_pct']:.1f}%"],
        ['Neutral Percentage', f"{metrics['neutral_pct']:.1f}%"],
        ['Average Confidence Score', f"{metrics['avg_confidence']:.3f}"],
        ['Average Polarity Score', f"{metrics['avg_polarity']:.3f}"],
        ['Average Subjectivity Score', f"{metrics['avg_subjectivity']:.3f}"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metrics_table)
    story.append(PageBreak())
    
    # Sample Data Table (first 10 rows)
    story.append(Paragraph("Sample Analysis Results", heading_style))
    
    # Prepare sample data
    sample_df = df.head(10)
    sample_data = [['Text (Truncated)', 'Sentiment', 'Confidence', 'Polarity']]
    
    for _, row in sample_df.iterrows():
        text_truncated = (row['text'][:50] + '...') if len(row['text']) > 50 else row['text']
        sample_data.append([
            text_truncated,
            row['sentiment'],
            f"{row['confidence']:.3f}",
            f"{row['polarity']:.3f}"
        ])
    
    sample_table = Table(sample_data, colWidths=[3*inch, 1*inch, 1*inch, 1*inch])
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    
    story.append(sample_table)
    story.append(Spacer(1, 20))
    
    # Methodology
    story.append(Paragraph("Methodology", heading_style))
    methodology_text = """
    This sentiment analysis was performed using TextBlob, a Python library for processing textual data. 
    The analysis includes:
    
    1. Sentiment Classification: Each text is classified as Positive, Negative, or Neutral based on polarity scores.
    2. Confidence Scoring: Confidence is derived from the absolute value of the polarity score.
    3. Keyword Extraction: Important keywords are extracted using NLTK tokenization and stop-word filtering.
    4. Polarity Score: Ranges from -1.0 (most negative) to 1.0 (most positive).
    5. Subjectivity Score: Ranges from 0.0 (objective) to 1.0 (subjective).
    """
    
    story.append(Paragraph(methodology_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data, filename

def batch_process_files(uploaded_files):
    """Process multiple uploaded files for batch analysis"""
    all_results = []
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    # Add file source information
                    df['file_source'] = uploaded_file.name
                    all_results.append(df)
            elif uploaded_file.name.endswith('.txt'):
                # Read text file line by line
                content = uploaded_file.read().decode('utf-8')
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                df = pd.DataFrame({
                    'text': lines,
                    'file_source': uploaded_file.name
                })
                all_results.append(df)
        except Exception as e:
            print(f"Error processing file {uploaded_file.name}: {str(e)}")
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

