import docx
import os

def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

file_path = 'RubixCubeSolver_Architecture.docx'
if os.path.exists(file_path):
    print(read_docx(file_path))
else:
    print(f"File {file_path} not found.")
