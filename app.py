import os
import io
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

# Import our custom utilities
from utils.predict import predict_disease
from utils.groq_helper import get_disease_info, chat_assistant

# ReportLab imports for generating PDF reports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def extract_list_items(html, heading_keyword):
    # Find heading matching heading_keyword and extract the ul block immediately following it
    pattern = re.compile(rf'<h3[^>]*>[^<]*{heading_keyword}[^<]*</h3>\s*<ul[^>]*>(.*?)</ul>', re.IGNORECASE | re.DOTALL)
    match = pattern.search(html)
    if not match:
        return []
    
    ul_content = match.group(1)
    li_pattern = re.compile(r'<li[^>]*>(.*?)</li>', re.IGNORECASE | re.DOTALL)
    items = li_pattern.findall(ul_content)
    
    cleaned_items = []
    for item in items:
        item_text = item.strip()
        item_text = item_text.replace('<strong>', '<b>').replace('</strong>', '</b>')
        # Remove any other custom tags if present (ReportLab Paragraph throws errors on unclosed or unsupported tags)
        item_text = re.sub(r'<(?!\/?(b|i|u|font|a))[^>]+>', '', item_text)
        cleaned_items.append(item_text)
    return cleaned_items

app = Flask(__name__)
app.secret_key = "skinsense_ai_secret_key"  # Required for flash messages

# Directory configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if a file was uploaded
    if 'file' not in request.files:
        flash("No file part in the request.")
        return redirect(url_for('index'))
        
    file = request.files['file']
    
    # 2. Check if user selected an empty file
    if file.filename == '':
        flash("No image selected for analysis.")
        return redirect(url_for('index'))
        
    # 3. Validate file extension
    if not allowed_file(file.filename):
        flash("Invalid file format. Supported formats: PNG, JPG, JPEG, WEBP.")
        return redirect(url_for('index'))
        
    try:
        # 4. Save file securely
        filename = secure_filename(file.filename)
        # Add timestamp to prevent caching issues and filename collisions
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 5. Run prediction
        print(f"[SkinSense AI] Running prediction for: {filepath}")
        result = predict_disease(filepath)
        
        # 6. Retrieve disease information using Groq (or fallback)
        disease_name = result['disease_name']
        confidence = result['confidence']
        all_predictions = result['all_predictions']
        
        print(f"[SkinSense AI] Predicted Condition: {disease_name} ({confidence}%)")
        disease_info_html = get_disease_info(disease_name)
        
        # Create image path relative to static/
        relative_image_path = f"uploads/{filename}"
        
        return render_template(
            'result.html',
            image_path=relative_image_path,
            filename=filename,
            disease_name=disease_name,
            confidence=confidence,
            is_confident=result.get('is_confident', True),
            warning=result.get('warning', ''),
            all_predictions=all_predictions,
            disease_info=disease_info_html
        )
        
    except Exception as e:
        print(f"[SkinSense AI] Error during prediction request: {e}")
        flash(f"An error occurred during analysis: {str(e)}")
        return redirect(url_for('index'))


@app.route('/download-report/<filename>')
def download_report(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    
    if not os.path.exists(filepath):
        flash("The requested image file could not be found.")
        return redirect(url_for('index'))
        
    try:
        # Re-run prediction to compile PDF details
        result = predict_disease(filepath)
        disease_name = result['disease_name']
        confidence = result['confidence']
        all_predictions = result['all_predictions']
        date_str = datetime.now().strftime("%B %d, %Y")
        time_str = datetime.now().strftime("%I:%M %p")
        
        # Set up a byte buffer for PDF generation
        buffer = io.BytesIO()
        
        # Define Document
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter, 
            rightMargin=45, 
            leftMargin=45, 
            topMargin=45, 
            bottomMargin=45
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles for the report
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=colors.HexColor('#085041'), # Dark Teal
            spaceAfter=15
        )
        
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#555555'),
            spaceAfter=25
        )
        
        h2_style = ParagraphStyle(
            'Heading2',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor('#1D9E75'), # Teal accent
            spaceAfter=10,
            spaceBefore=15
        )
        
        body_style = ParagraphStyle(
            'BodyTextCustom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=8
        )
        
        disclaimer_style = ParagraphStyle(
            'DisclaimerCustom',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor('#C0392B'), # Deep Red
            spaceBefore=30
        )
        
        story = []
        
        # Header / Logo element
        story.append(Paragraph("🔬 SKINSENSE AI - CLINICAL ANALYSIS REPORT", title_style))
        story.append(Paragraph(f"Generated on: {date_str} at {time_str} | Source File: {filename}", subtitle_style))
        
        # Divider Line
        d_table = Table([[""]], colWidths=[520])
        d_table.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 2, colors.HexColor('#1D9E75')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0)
        ]))
        story.append(d_table)
        story.append(Spacer(1, 15))
        
        # Top half: Image & Core Metrics side by side in a table
        try:
            # Resize image for PDF representation (maintaining aspect ratio approx)
            report_img = RLImage(filepath, width=150, height=150)
        except Exception:
            report_img = Paragraph("[Image Preview Unavailable]", body_style)
            
        # Compile patient classification results text
        results_text = f"""
        <b>Primary Classification:</b><br/>
        <font size="14" color="#085041"><b>{disease_name}</b></font><br/><br/>
        <b>AI Confidence:</b><br/>
        <font size="14" color="#1D9E75"><b>{confidence:.2f}%</b></font><br/><br/>
        <b>Analysis Method:</b> Multimodal AI Vision Analysis<br/>
        <b>Model Architecture:</b> Llama 4 Scout Vision
        """
        
        results_paragraph = Paragraph(results_text, body_style)
        
        # Combine image & results into a table
        info_table_data = [[report_img, results_paragraph]]
        info_table = Table(info_table_data, colWidths=[180, 340])
        info_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BACKGROUND', (1,0), (1,0), colors.HexColor('#F4F6F6')),
            ('PADDING', (1,0), (1,0), 12),
            ('BOX', (1,0), (1,0), 1, colors.HexColor('#E5E8E8')),
            ('ALIGN', (0,0), (0,0), 'CENTER')
        ]))
        story.append(info_table)
        story.append(Spacer(1, 15))
        
        # Alternative Predictions Table
        story.append(Paragraph("Alternative Differential Diagnostics", h2_style))
        pred_rows = [["Classification Category", "Confidence Score"]]
        for pred in all_predictions[:4]:
            pred_rows.append([pred['label'], f"{pred['score']:.2f}%"])
            
        pred_table = Table(pred_rows, colWidths=[300, 220])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#085041')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (1,0), (1,-1), 'RIGHT'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8F9F9')]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 15))
        
        # Get disease information
        disease_info_html = get_disease_info(disease_name)
        
        allopathy_items = extract_list_items(disease_info_html, "Allopathy")
        homeopathy_items = extract_list_items(disease_info_html, "Homeopathy")
        siddha_items = extract_list_items(disease_info_html, "Siddha")
        
        if allopathy_items or homeopathy_items or siddha_items:
            story.append(Paragraph("Recommended Treatment Suggestions", h2_style))
            story.append(Spacer(1, 8))
            
            subheading_style = ParagraphStyle(
                'SubheadingCustom',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=colors.HexColor('#085041'),
                spaceAfter=4,
                spaceBefore=6
            )
            
            bullet_style = ParagraphStyle(
                'BulletCustom',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=9.5,
                leading=13,
                leftIndent=15,
                firstLineIndent=-10,
                spaceAfter=4
            )
            
            if allopathy_items:
                story.append(Paragraph("• Allopathy Suggestions", subheading_style))
                for item in allopathy_items:
                    story.append(Paragraph(f"▪ {item}", bullet_style))
                story.append(Spacer(1, 5))
                
            if homeopathy_items:
                story.append(Paragraph("• Homeopathy Suggestions", subheading_style))
                for item in homeopathy_items:
                    story.append(Paragraph(f"▪ {item}", bullet_style))
                story.append(Spacer(1, 5))
                
            if siddha_items:
                story.append(Paragraph("• Siddha Suggestions", subheading_style))
                for item in siddha_items:
                    story.append(Paragraph(f"▪ {item}", bullet_style))
                story.append(Spacer(1, 5))
                
            story.append(Spacer(1, 10))
            
        # Disclaimer block
        disclaimer_text = (
            "<b>⚠️ DISCLAIMER:</b> This report is generated automatically by a multimodal AI vision algorithm (Llama 4 Scout) "
            "and is intended solely for educational, informational, and triage assistance. "
            "It does NOT constitute medical advice, diagnosis, or treatment. Machine learning models are subject to statistical errors. "
            "Please consult a registered healthcare professional or dermatologist for clinical examination and proper diagnosis."
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build document
        doc.build(story)
        
        # Stream the buffer back to Flask
        buffer.seek(0)
        return send_file(
            buffer, 
            as_attachment=True, 
            download_name=f"SkinSense_Report_{disease_name.replace(' ', '_')}.pdf", 
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"[SkinSense AI] Error generating PDF report: {e}")
        flash("An error occurred while creating the PDF report.")
        return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        disease_name = data.get("disease_name", "").strip()
        
        if not user_message:
            return {"response": "Message cannot be empty."}, 400
            
        response_text = chat_assistant(user_message, disease_name)
        return {"response": response_text}
    except Exception as e:
        print(f"[SkinSense AI] Chat route error: {e}")
        return {"response": "An unexpected error occurred. Please try again later."}, 500

if __name__ == '__main__':
    # Flask will bind to all network interfaces on port 5000
    app.run(debug=True, port=5000)
