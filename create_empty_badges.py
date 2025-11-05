from PIL import Image

from add_text import add_crop_marks, images_to_pdf_2up_landscape

base_image = Image.open("badge.png")

images = [base_image, base_image]

pdf_buffer = images_to_pdf_2up_landscape(images, margin_in=0.25)

pdf_with_marks = add_crop_marks(
    pdf_buffer,
    mark_len=1000,       # make the guides longer
    mark_thickness=0.5,  # slightly thicker lines
    margin_in=0.25,    # same margin used in images_to_pdf_2up_landscape
    gap_in=0.25        # same gap between badges
)

with open("cards_2up_blank.pdf", "wb") as f:
    f.write(pdf_with_marks.getvalue())

