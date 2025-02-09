import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from add_text import (
    add_text_to_image,
    apply_name_affiliation_image,
    apply_texts_to_image,
    create_pdf_name_affiliation,
    images_to_pdf,
)

# Inject custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 500px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)


example_name = "John Cobra"
example_affiliation = "Test Affiliation"

# Streamlit App Title
st.title("Label generator")

# Create a DataFrame with 2 rows and 3 columns, including names for rows and columns
df_formatting = pd.DataFrame(
    {
        "Font size, px": [80, 50],
        "Offset x, %": [0, 0],
        "Offset y, %": [0, -20],
    },
    index=["Names", "Affiliations"],
)

df_print_to_pdf = pd.DataFrame(
    {
        "Rows": [3],
        "Columns": [2],
    },
    index=["Number"],
)

# Display the editable dataframe with row and column names
edited_df_formatting = st.sidebar.data_editor(df_formatting)
edited_df_to_pdf = st.sidebar.data_editor(df_print_to_pdf)

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# File Upload - Image or PDF
uploaded_card = st.sidebar.file_uploader(
    "Upload an Image", type=["png", "jpg", "jpeg", "pdf"]
)
if uploaded_card is not None:
    st.write("### Uploaded File Details:")
    # st.write(f"File Name: {uploaded_card.name}")
    base_image = Image.open(uploaded_card)
    image_to_show = apply_name_affiliation_image(
        base_image,
        df_formatting=edited_df_formatting,
        name=example_name,
        affiliation=example_affiliation,
    )
    st.image(image_to_show, caption="Uploaded Image", use_container_width=True)


if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Print number of rows
    st.write(f"### Number of Atendees: {df.shape[0]}")

    # Display the DataFrame
    st.write("### Data Preview:")
    st.dataframe(df)

if uploaded_file is not None and uploaded_card is not None:

    st.download_button(
        label="Download image",
        data=create_pdf_name_affiliation(base_image, df_formatting, df, edited_df_to_pdf),
        file_name="output.pdf",
        mime="application/pdf",
    )
