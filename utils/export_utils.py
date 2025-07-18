from fpdf import FPDF
import pandas as pd
import io

def export_to_pdf(data, filename="passport_data.pdf"):
    # Remove validation data and debug fields for export
    export_data = {k: v for k, v in data.items() if k not in ["_validation", "_ocr_text"]}
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Passport OCR Extracted Info", ln=True, align='C')
    pdf.ln(10)
    
    # Add validation summary if available
    if "Validation Issues" in export_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"Validation Issues: {export_data['Validation Issues']}", ln=True)
        
        if "Critical Issues" in export_data:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(200, 10, txt=f"Critical Issues: {export_data['Critical Issues']}", ln=True)
        
        pdf.ln(5)
    
    # Add the rest of the data
    pdf.set_font("Arial", size=12)
    for key, value in export_data.items():
        if key not in ["Validation Issues", "Critical Issues"]:
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    output = io.BytesIO()
    pdf.output(output)
    return output.getvalue()

def export_to_excel(data):
    # If it's a batch, handle differently
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        # This is a batch of documents
        rows = []
        for filename, doc_data in data.items():
            # Remove validation data but keep summary
            row = {k: v for k, v in doc_data.items() if k != "_validation"}
            row["Filename"] = filename
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        # Single document
        export_data = {k: v for k, v in data.items() if k != "_validation"}
        df = pd.DataFrame([export_data])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    
    return output.getvalue()
