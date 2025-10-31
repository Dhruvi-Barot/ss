"""
PDF report generation module
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO
import pandas as pd
from datetime import datetime
import streamlit as st


def generate_pdf_report(ticker: str, analysis_data: dict) -> BytesIO:
    """
    Generate professional PDF report
    
    Args:
        ticker: Stock ticker symbol
        analysis_data: Dictionary with all analysis data
    
    Returns:
        BytesIO object with PDF content
    """
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00D9FF'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#00D9FF'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY
        )
        
        # Title
        story.append(Paragraph(f"📈 Stock Analysis Report: {ticker}", title_style))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y at %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        current_price = analysis_data.get('current_price', 0)
        predicted_price = analysis_data.get('predicted_price', 0)
        change_pct = ((predicted_price - current_price) / current_price * 100) if current_price else 0
        
        summary_table_data = [
            ['Metric', 'Value'],
            ['Current Price', f"₹{current_price:.2f}"],
            ['Predicted Price (Next Day)', f"₹{predicted_price:.2f}"],
            ['Expected Change', f"{change_pct:+.2f}%"],
            ['Recommendation', analysis_data.get('recommendation', 'N/A')],
            ['Model Confidence', f"{analysis_data.get('confidence', 0):.1f}%"]
        ]
        
        summary_table = Table(summary_table_data, colWidths=[3*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00D9FF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Company Information
        story.append(Paragraph("Company Information", heading_style))
        
        company_info = analysis_data.get('company_info', {})
        
        company_table_data = [
            ['Information', 'Details'],
            ['Company Name', company_info.get('company_name', 'N/A')],
            ['Sector', company_info.get('sector', 'N/A')],
            ['Industry', company_info.get('industry', 'N/A')],
            ['Market Cap', f"₹{company_info.get('market_cap', 0)/1e9:.2f}B"]
        ]
        
        company_table = Table(company_table_data, colWidths=[2.5*inch, 3*inch])
        company_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00D9FF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey)
        ]))
        
        story.append(company_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Financial Analysis
        story.append(Paragraph("Financial Analysis", heading_style))
        
        fundamentals = company_info.get('fundamentals', {})
        
        financial_table_data = [
            ['Metric', 'Value'],
            ['P/E Ratio', f"{fundamentals.get('pe_ratio', 'N/A')}"],
            ['ROE (Return on Equity)', f"{fundamentals.get('roe', 'N/A')}%"],
            ['Profit Margin', f"{fundamentals.get('profit_margin', 'N/A')}%"],
            ['Debt to Equity', f"{fundamentals.get('debt_to_equity', 'N/A')}"],
            ['Beta', f"{fundamentals.get('beta', 'N/A')}"],
            ['Dividend Yield', f"{fundamentals.get('dividend_yield', 'N/A')}%"],
            ['52-Week High', f"₹{fundamentals.get('52_week_high', 'N/A')}"],
            ['52-Week Low', f"₹{fundamentals.get('52_week_low', 'N/A')}"]
        ]
        
        financial_table = Table(financial_table_data, colWidths=[3*inch, 2.5*inch])
        financial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00D9FF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey)
        ]))
        
        story.append(financial_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Technical Analysis
        story.append(Paragraph("Technical Analysis", heading_style))
        
        technical = analysis_data.get('technical', {})
        
        tech_text = f"""
        <b>RSI (14):</b> {technical.get('rsi', 'N/A')} {'- Overbought' if technical.get('rsi', 0) > 70 else '- Oversold' if technical.get('rsi', 0) < 30 else '- Neutral'}<br/>
        <b>MACD:</b> {technical.get('macd', 'N/A')} {'- Bullish' if technical.get('macd', 0) > technical.get('macd_signal', 0) else '- Bearish'}<br/>
        <b>Moving Averages:</b> SMA 50: ₹{technical.get('sma_50', 'N/A')} | SMA 200: ₹{technical.get('sma_200', 'N/A')}<br/>
        <b>Trend:</b> {technical.get('trend', 'N/A')}
        """
        
        story.append(Paragraph(tech_text, normal_style))
        story.append(Spacer(1, 0.3*inch))
        
        # News Sentiment Analysis
        story.append(Paragraph("News Sentiment Analysis", heading_style))
        
        sentiment = analysis_data.get('sentiment', {})
        
        sentiment_text = f"""
        <b>Overall Sentiment:</b> {sentiment.get('sentiment_label', 'N/A')}<br/>
        <b>Sentiment Score:</b> {sentiment.get('sentiment_score', 0):.2f} (-1 to +1)<br/>
        <b>Positive News:</b> {sentiment.get('positive_ratio', 0)*100:.1f}%<br/>
        <b>Negative News:</b> {sentiment.get('negative_ratio', 0)*100:.1f}%<br/>
        <b>Neutral News:</b> {sentiment.get('neutral_ratio', 0)*100:.1f}%<br/>
        <b>Total Articles Analyzed:</b> {sentiment.get('article_count', 0)}
        """
        
        story.append(Paragraph(sentiment_text, normal_style))
        story.append(Spacer(1, 0.3*inch))
        
        # AI Prediction Explanation
        story.append(PageBreak())
        story.append(Paragraph("AI Prediction Explanation", heading_style))
        
        explanation = analysis_data.get('explanation', 'No explanation available')
        story.append(Paragraph(explanation, normal_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Risk Disclaimer
        story.append(Paragraph("Important Disclaimer", heading_style))
        
        disclaimer = Paragraph(
            "<b>Disclaimer:</b> This report is generated for informational and educational purposes only. "
            "It is NOT financial advice. The predictions made by the AI model are based on historical data "
            "and machine learning algorithms, which may not accurately predict future stock performance. "
            "Stock market investments carry risk of loss. Please consult with a certified financial advisor "
            "before making any investment decisions. Past performance does not guarantee future results.",
            normal_style
        )
        story.append(disclaimer)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None


def download_pdf_button(pdf_buffer: BytesIO, filename: str):
    """
    Create Streamlit download button for PDF
    
    Args:
        pdf_buffer: BytesIO object with PDF content
        filename: Filename for download
    """
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name=filename,
        mime="application/pdf"
    )
