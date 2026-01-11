from docx import Document

def generate_document(path: str, text: str):
    """Generate a Word document (.docx) with the given text content."""
    try:
        doc = Document()
        doc.add_paragraph(text)
        doc.save(path)
        return {"path": path, "status": "created"}
    except Exception as e:
        return {"error": str(e)}

# Tool export for dynamic loading
TOOL = {
    "name": "doc_generator",
    "func": generate_document,
    "description": "Generate a Word document (.docx) with text content. Study platform use: create study notes, summaries, reports."
}
