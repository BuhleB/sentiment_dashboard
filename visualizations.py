import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import io
from datetime import datetime, timedelta
import numpy as np

def create_sentiment_distribution_chart(df):
    """Create a pie chart showing sentiment distribution"""
    if df.empty:
        return None
    
    sentiment_counts = df['sentiment'].value_counts()
    
    # Define colors for sentiments
    colors = {
        'Positive': '#2E8B57',  # Sea Green
        'Negative': '#DC143C',  # Crimson
        'Neutral': '#4682B4'    # Steel Blue
    }
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_sentiment_over_time_chart(df):
    """Create a line chart showing sentiment trends over time"""
    if df.empty or 'date' not in df.columns:
        return None
    
    # Convert date column to datetime
    df_copy = df.copy()
    try:
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    except:
        # If date conversion fails, create a simple index-based chart
        df_copy['date'] = range(len(df_copy))
    
    # Group by date and sentiment
    sentiment_over_time = df_copy.groupby(['date', 'sentiment']).size().reset_index(name='count')
    
    # Create line chart
    fig = px.line(
        sentiment_over_time,
        x='date',
        y='count',
        color='sentiment',
        title="Sentiment Trends Over Time",
        color_discrete_map={
            'Positive': '#2E8B57',
            'Negative': '#DC143C',
            'Neutral': '#4682B4'
        }
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Texts",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_sentiment_by_source_chart(df):
    """Create a bar chart showing sentiment distribution by source"""
    if df.empty or 'source' not in df.columns:
        return None
    
    # Group by source and sentiment
    source_sentiment = df.groupby(['source', 'sentiment']).size().reset_index(name='count')
    
    fig = px.bar(
        source_sentiment,
        x='source',
        y='count',
        color='sentiment',
        title="Sentiment Distribution by Source",
        color_discrete_map={
            'Positive': '#2E8B57',
            'Negative': '#DC143C',
            'Neutral': '#4682B4'
        },
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title="Source",
        yaxis_title="Number of Texts",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_confidence_distribution_chart(df):
    """Create a histogram showing confidence score distribution"""
    if df.empty or 'confidence' not in df.columns:
        return None
    
    fig = px.histogram(
        df,
        x='confidence',
        color='sentiment',
        title="Confidence Score Distribution",
        nbins=20,
        color_discrete_map={
            'Positive': '#2E8B57',
            'Negative': '#DC143C',
            'Neutral': '#4682B4'
        }
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        height=400,
        bargap=0.1
    )
    
    return fig

def create_polarity_vs_subjectivity_scatter(df):
    """Create a scatter plot of polarity vs subjectivity"""
    if df.empty or 'polarity' not in df.columns or 'subjectivity' not in df.columns:
        return None
    
    fig = px.scatter(
        df,
        x='polarity',
        y='subjectivity',
        color='sentiment',
        title="Polarity vs Subjectivity",
        hover_data=['text'],
        color_discrete_map={
            'Positive': '#2E8B57',
            'Negative': '#DC143C',
            'Neutral': '#4682B4'
        }
    )
    
    fig.update_layout(
        xaxis_title="Polarity (Negative ← → Positive)",
        yaxis_title="Subjectivity (Objective ← → Subjective)",
        height=400
    )
    
    # Add quadrant lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def create_wordcloud(df, sentiment_filter=None):
    """Create a word cloud from keywords"""
    if df.empty or 'keywords' not in df.columns:
        return None
    
    # Filter by sentiment if specified
    if sentiment_filter:
        df_filtered = df[df['sentiment'].isin(sentiment_filter)]
    else:
        df_filtered = df
    
    # Collect all keywords
    all_keywords = []
    for keywords_list in df_filtered['keywords']:
        if isinstance(keywords_list, list):
            all_keywords.extend(keywords_list)
        elif isinstance(keywords_list, str):
            # Handle case where keywords might be stored as string
            try:
                import ast
                keywords_list = ast.literal_eval(keywords_list)
                all_keywords.extend(keywords_list)
            except:
                all_keywords.extend(keywords_list.split(','))
    
    if not all_keywords:
        return None
    
    # Count keyword frequencies
    keyword_freq = Counter(all_keywords)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate_from_frequencies(keyword_freq)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Common Keywords', fontsize=16, fontweight='bold')
    
    return fig

def create_keyword_frequency_chart(df, top_n=10):
    """Create a bar chart of most frequent keywords"""
    if df.empty or 'keywords' not in df.columns:
        return None
    
    # Collect all keywords
    all_keywords = []
    for keywords_list in df['keywords']:
        if isinstance(keywords_list, list):
            all_keywords.extend(keywords_list)
        elif isinstance(keywords_list, str):
            try:
                import ast
                keywords_list = ast.literal_eval(keywords_list)
                all_keywords.extend(keywords_list)
            except:
                all_keywords.extend(keywords_list.split(','))
    
    if not all_keywords:
        return None
    
    # Count keyword frequencies
    keyword_freq = Counter(all_keywords)
    top_keywords = keyword_freq.most_common(top_n)
    
    if not top_keywords:
        return None
    
    keywords, frequencies = zip(*top_keywords)
    
    fig = px.bar(
        x=list(frequencies),
        y=list(keywords),
        orientation='h',
        title=f"Top {top_n} Most Frequent Keywords",
        labels={'x': 'Frequency', 'y': 'Keywords'}
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_sentiment_metrics_summary(df):
    """Create summary metrics for sentiment analysis"""
    if df.empty:
        return {}
    
    total_texts = len(df)
    
    # Sentiment counts
    sentiment_counts = df['sentiment'].value_counts()
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    
    # Percentages
    positive_pct = (positive_count / total_texts) * 100 if total_texts > 0 else 0
    negative_pct = (negative_count / total_texts) * 100 if total_texts > 0 else 0
    neutral_pct = (neutral_count / total_texts) * 100 if total_texts > 0 else 0
    
    # Average confidence and polarity
    avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0
    avg_polarity = df['polarity'].mean() if 'polarity' in df.columns else 0
    avg_subjectivity = df['subjectivity'].mean() if 'subjectivity' in df.columns else 0
    
    return {
        'total_texts': total_texts,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct,
        'neutral_pct': neutral_pct,
        'avg_confidence': avg_confidence,
        'avg_polarity': avg_polarity,
        'avg_subjectivity': avg_subjectivity
    }

def display_comparative_analysis(df1, df2, label1="Dataset 1", label2="Dataset 2"):
    """Display comparative analysis between two datasets"""
    if df1.empty or df2.empty:
        st.warning("Both datasets must have data for comparative analysis.")
        return
    
    st.subheader("📊 Comparative Analysis")
    
    # Create comparison metrics
    metrics1 = create_sentiment_metrics_summary(df1)
    metrics2 = create_sentiment_metrics_summary(df2)
    
    # Display comparison table
    comparison_data = {
        'Metric': [
            'Total Texts',
            'Positive Count',
            'Negative Count', 
            'Neutral Count',
            'Positive %',
            'Negative %',
            'Neutral %',
            'Avg Confidence',
            'Avg Polarity',
            'Avg Subjectivity'
        ],
        label1: [
            metrics1['total_texts'],
            metrics1['positive_count'],
            metrics1['negative_count'],
            metrics1['neutral_count'],
            f"{metrics1['positive_pct']:.1f}%",
            f"{metrics1['negative_pct']:.1f}%",
            f"{metrics1['neutral_pct']:.1f}%",
            f"{metrics1['avg_confidence']:.3f}",
            f"{metrics1['avg_polarity']:.3f}",
            f"{metrics1['avg_subjectivity']:.3f}"
        ],
        label2: [
            metrics2['total_texts'],
            metrics2['positive_count'],
            metrics2['negative_count'],
            metrics2['neutral_count'],
            f"{metrics2['positive_pct']:.1f}%",
            f"{metrics2['negative_pct']:.1f}%",
            f"{metrics2['neutral_pct']:.1f}%",
            f"{metrics2['avg_confidence']:.3f}",
            f"{metrics2['avg_polarity']:.3f}",
            f"{metrics2['avg_subjectivity']:.3f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Side-by-side sentiment distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = create_sentiment_distribution_chart(df1)
        if fig1:
            fig1.update_layout(title=f"Sentiment Distribution - {label1}")
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = create_sentiment_distribution_chart(df2)
        if fig2:
            fig2.update_layout(title=f"Sentiment Distribution - {label2}")
            st.plotly_chart(fig2, use_container_width=True)

