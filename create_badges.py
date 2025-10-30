import pandas as pd
from PIL import Image

from add_text import create_pdf_name_affiliation, images_to_pdf_2up_landscape

base_image = Image.open("badge.png")

# First number is for names, second for affiliations
df_formatting = pd.DataFrame(
        {
            "Font size, px": [120, 75],
            "Offset x, %": [0, 0],
            "Offset y, %": [-30, -40],
        },
        index=["Names", "Affiliations"],
    )

df_names_affiliations = pd.read_csv("data.csv")

images = create_pdf_name_affiliation(
    base_image, df_formatting, df_names_affiliations
)

pdf_buffer = images_to_pdf_2up_landscape(images, margin_in=0)
with open("cards_2up.pdf", "wb") as f:
    f.write(pdf_buffer.getvalue())