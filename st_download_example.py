import io
from io import BytesIO

import streamlit as st


def open_pdf_to_memory(pdf_path):
    """Opens a PDF from the given path into memory.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        io.BytesIO: A BytesIO object containing the PDF data in memory, or None if an error occurs.
    """
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
            pdf_buffer = io.BytesIO(pdf_data)
            return pdf_buffer
    except FileNotFoundError:
        print(f"Error: File not found: {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Create the PDF in memory
pdf_data = open_pdf_to_memory("output.pdf")

# Display the download button
st.title("PDF File Download Example")
st.write("Click the button below to download the PDF file.")

# Provide the in-memory PDF for download
st.download_button(
    label="Download PDF",
    data=pdf_data,
    file_name="sample_pdf.pdf",
    mime="application/pdf",
)
