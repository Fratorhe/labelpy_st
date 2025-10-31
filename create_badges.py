import pandas as pd
from PIL import Image

from add_text import create_pdf_name_affiliation, images_to_pdf_2up_landscape, add_crop_marks

base_image = Image.open("badge.png")

# First number is for names, second for affiliations
df_formatting = pd.DataFrame(
        {
            "Font size, px": [100, 60],
            "Offset x, %": [0, 0],
            "Offset y, %": [-27, -40],
        },
        index=["Names", "Affiliations"],
    )

df_names_affiliations = pd.read_csv("attendees.csv",sep='\t')

# Map your CSV’s columns to the expected ones
column_map = {
    "NAME": "Name",
    "INSTITUTION": "Institution"
}

# Rename only if the expected columns are missing
df_names_affiliations.rename(columns=column_map, inplace=True)

images = create_pdf_name_affiliation(
     base_image, df_formatting, df_names_affiliations
 )

# pdf_buffer = images_to_pdf_2up_landscape(images, margin_in=0)

# pdf_with_marks = add_crop_marks(pdf_buffer, mark_len=25, mark_thickness=0.7)

# with open("cards_2up.pdf", "wb") as f:
#     f.write(pdf_with_marks.getvalue())

pdf_buffer = images_to_pdf_2up_landscape(images, margin_in=0.25)

pdf_with_marks = add_crop_marks(
    pdf_buffer,
    mark_len=1000,       # make the guides longer
    mark_thickness=1,  # slightly thicker lines
    margin_in=0.25,    # same margin used in images_to_pdf_2up_landscape
    gap_in=0.25        # same gap between badges
)

with open("cards_2up.pdf", "wb") as f:
    f.write(pdf_with_marks.getvalue())

