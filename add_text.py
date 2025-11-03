import io
import os
import tempfile

import fpdf
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch, cm
from reportlab.lib.utils import ImageReader
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter




def add_text_to_image(
    image,
    text,
    offset=(0, 0),
    font_path=None,
    font_size=200,
    text_color=(255, 0, 0),
    save=False,
    max_width_ratio=0.9,
):
    """
    Adds centered text to an image with offsets given in % of width/height.
    Positive x moves right. Positive y moves UP (per your docstring).
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # percent offsets -> pixels
    offset_x = (offset[0] * width) / 100.0
    offset_y = (offset[1] * height) / 100.0

    # load font (use provided font_path if given)
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # fallback to a common font if available, else default bitmap
            font = ImageFont.truetype("./fonts/OpenSans2.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # wrapping text
    max_text_width = width * max_width_ratio
    words = text.split()
    lines = []
    current_line = words[0]

    for word in words[1:]:
        line_test = current_line + " " + word
        if draw.textlength(line_test, font=font) <= max_text_width:
            current_line = line_test
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    # compute total height for centering
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 20
    total_text_height = line_height * len(lines)

    # starting y (centered vertically)
    y = (height - total_text_height) / 2.0 - offset_y

    # draw each line centered
    for line in lines:
        line_width = draw.textlength(line, font=font)
        x = (width - line_width) / 2.0 + offset_x
        draw.text((x, y), line, fill=text_color, font=font)
        y += line_height

    if save:
        image.save("output.jpg")

    return image

    # # text bbox for centering
    # bbox = draw.textbbox((0, 0), text, font=font)
    # text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # # center, then apply offsets (y positive up means subtract)
    # x = (width - text_width) / 2.0 + offset_x
    # y = (height - text_height) / 2.0 - offset_y

    # draw.text((x, y), text, fill=text_color, font=font)

    # if save:
    #     image.save("output.jpg")

    # return image

def apply_name_affiliation_image(base_image, df_formatting, name, affiliation):
    """
    Applies a name and affiliation to a base image using formatting details from a DataFrame.

    :param base_image: PIL Image object. The original image to which the text will be added.
    :param df_formatting: pandas DataFrame. A DataFrame containing the formatting details for names and affiliations,
                           such as font size and text offsets.
    :param name: str. The name to be added to the image.
    :param affiliation: str. The affiliation to be added to the image.

    :return: PIL Image object. The modified image with the name and affiliation applied.
    """
    # Create a copy of the base image to avoid altering the original image
    img_copy = base_image.copy()

    # Extract the formatting details for names from the DataFrame
    font_size_names = df_formatting.loc["Names", "Font size, px"]
    offset_x_names = df_formatting.loc["Names", "Offset x, %"]
    offset_y_names = df_formatting.loc["Names", "Offset y, %"]

    # Extract the formatting details for affiliations from the DataFrame
    font_size_affiliation = df_formatting.loc["Affiliations", "Font size, px"]
    offset_x_affiliation = df_formatting.loc["Affiliations", "Offset x, %"]
    offset_y_affiliation = df_formatting.loc["Affiliations", "Offset y, %"]

    # Add the name to the image using the extracted formatting details
    modified_image = add_text_to_image(
        img_copy,
        name,
        font_size=font_size_names,
        offset=(offset_x_names, offset_y_names),
        text_color=(0, 0, 0),
    )

    # Add the affiliation to the image using the extracted formatting details
    modified_image = add_text_to_image(
        modified_image,
        affiliation,
        font_size=font_size_affiliation,
        offset=(offset_x_affiliation, offset_y_affiliation),
        text_color=(0, 0, 0),
    )
    return modified_image


def apply_texts_to_image(
    image_path, texts, font_path=None, font_size=200, text_color=(255, 0, 0)
):
    """
    Applies multiple text variations to the same image and returns them in memory.

    :param image_path: Path to the input image.
    :param texts: List of texts to add to the image.
    :param font_path: Path to a .ttf font file (default is None, which uses a system default font).
    :param font_size: Font size.
    :param text_color: Tuple (R, G, B) representing text color.
    """
    base_image = Image.open(image_path)
    images = []

    for text in texts:
        img_copy = base_image.copy()
        modified_image = add_text_to_image(
            img_copy,
            text,
            font_path=font_path,
            font_size=font_size,
            text_color=text_color,
        )
        images.append(modified_image)

    return images


from PIL import Image


# def images_to_pdf(images, rows, columns, dpi=300, page_margin_px=20, gutter_px=10):
#     """
#     Compose images into a grid on letter-sized pages using PIL only.
#     Returns a BytesIO PDF stream.
#     """
#     if not images:
#         raise ValueError("No images to convert to PDF.")

#     # Page size in pixels at the chosen dpi (8.5x11 in)
#     page_w = int(8.5 * dpi)
#     page_h = int(11.0 * dpi)
#     page_size = (page_w, page_h)

#     imgs_per_row = max(1, int(columns))
#     imgs_per_col = max(1, int(rows))
#     imgs_per_page = imgs_per_row * imgs_per_col

#     # Compute drawable area and per-cell max size
#     drawable_w = page_w - 2 * page_margin_px - (imgs_per_row - 1) * gutter_px
#     drawable_h = page_h - 2 * page_margin_px - (imgs_per_col - 1) * gutter_px
#     cell_w = drawable_w // imgs_per_row
#     cell_h = drawable_h // imgs_per_col

#     pdf_pages = []
#     page = Image.new("RGB", page_size, "white")

#     def cell_origin(idx_within_page):
#         r = idx_within_page // imgs_per_row
#         c = idx_within_page % imgs_per_row
#         x = page_margin_px + c * (cell_w + gutter_px)
#         y = page_margin_px + r * (cell_h + gutter_px)
#         return x, y

#     j = 0  # index within current page

#     for img in images:
#         # ensure RGB and copy before thumbnail so we don't mutate original
#         work = img.convert("RGB").copy()
#         work.thumbnail((cell_w, cell_h))  # in-place, preserves aspect

#         x, y = cell_origin(j)
#         # center inside the cell
#         paste_x = x + (cell_w - work.width) // 2
#         paste_y = y + (cell_h - work.height) // 2
#         page.paste(work, (paste_x, paste_y))

#         j += 1
#         if j == imgs_per_page:
#             pdf_pages.append(page)
#             page = Image.new("RGB", page_size, "white")
#             j = 0

#     # flush last partially filled page
#     if j != 0:
#         pdf_pages.append(page)

#     # Save all pages to a single PDF in-memory
#     buf = io.BytesIO()
#     # Pillow can save multi-page PDFs by passing the remaining images via append_images
#     pdf_pages[0].save(buf, format="PDF", save_all=True, append_images=pdf_pages[1:])
#     buf.seek(0)
#     return buf

def create_pdf_name_affiliation(
    base_image, df_formatting, df_names_affiliations
):
    """
    Creates a PDF containing images with names and affiliations added to a base image.

    :param base_image: PIL Image object. The base image to which names and affiliations will be added.
    :param df_formatting: pandas DataFrame. Contains the formatting details for the text (e.g., font size, offset).
    :param df_names_affiliations: pandas DataFrame. Contains the names and affiliations to be added to the base image.
    :param df_to_pdf_options: pandas DataFrame. Contains options for the PDF layout (e.g., number of rows and columns).

    :return: PDF buffer. The generated PDF containing the modified images.
    """
    # creates the images and adds them to the container
    images = []
    for index, row in df_names_affiliations.iterrows():
        name = row.Name.title() # title makes format as: Name Last Name
        affiliation = row.Institution
        modified_image = apply_name_affiliation_image(
            base_image, df_formatting, name=name, affiliation=affiliation
        )
        images.append(modified_image)

    # Convert the list of images into a PDF buffer
    # pdf_buffer = images_to_pdf(images, rows=rows, columns=columns)
    return images


def images_to_pdf_2up_landscape(images, margin_in=1.0, gutter_in=0.25):
    """
    Create a 2-up (two per page) PDF on US-letter landscape (11 x 8.5 in),
    placing each image at exactly 4 x 6 inches (W x H). Uses FPDF to embed
    PNGs losslessly (no recompression). Returns a BytesIO buffer.
    """
    if not images:
        raise ValueError("No images provided")

    # Page setup in inches
    page_w, page_h = 11.0, 8.5
    card_w, card_h = 4.0, 6.0  # your images are 4x6 (W x H) inches
    # Compute horizontal placement
    total_cards_w = (card_w * 2) + gutter_in
    if total_cards_w + 2 * margin_in > page_w or card_h + 2 * margin_in > page_h:
        raise ValueError("Margins/gutter too large to fit two 4x6 cards on the page.")

    left_x = (page_w - total_cards_w) / 2.0
    right_x = left_x + card_w + gutter_in
    y_pos = (page_h - card_h) / 2.0  # vertical centering

    # Write each image to a temporary PNG file (lossless)
    tmp_paths = []
    try:
        for im in images:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_paths.append(tmp.name)
            im.convert("RGB").save(tmp.name, format="PNG")  # no recompression loss
            tmp.close()

        pdf = fpdf.FPDF(orientation="L", unit="in", format="Letter")
        # Add images in pairs per page
        for i in range(0, len(tmp_paths), 2):
            pdf.add_page()
            # Left image
            pdf.image(tmp_paths[i], x=left_x, y=y_pos, w=card_w, h=card_h)
            # Right image, if present
            if i + 1 < len(tmp_paths):
                pdf.image(tmp_paths[i + 1], x=right_x, y=y_pos, w=card_w, h=card_h)

        # Export to BytesIO
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        buf = io.BytesIO(pdf_bytes)
        buf.seek(0)
        return buf
    finally:
        # Clean up temporaries
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

if __name__ == "__main__":
    image_name = "card_example.jpg"
    base_image = Image.open("card_example.jpg")

    my_list = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
    ]

    # Example usage
    images = apply_texts_to_image(
        "badge.png",
        my_list,
        font_size=150,
        text_color=(255, 0, 0),
    )
    pdf_buffer = images_to_pdf(images, rows=2, columns=2)

    df_formatting = pd.DataFrame(
        {
            "Font size, px": [50, 30],
            "Offset x, %": [0, 20],
            "Offset y, %": [0, -20],
        },
        index=["Names", "Affiliations"],
    )
    df_names_affiliations = pd.DataFrame(
        {
            "Name": ["Fran", "Luis", "asfd"],
            "Institution": ["uiu", "df", "dfdf"],
        },
    )
    df_to_pdf_options = pd.DataFrame(
        {
            "Rows": [3],
            "Columns": [2],
        },
        index=["Number"],
    )
    pdf_buffer = create_pdf_name_affiliation(
        base_image, df_formatting, df_names_affiliations
    )

    # Optionally, save the PDF to a file
    with open("output_test.pdf", "wb") as f:
        f.write(pdf_buffer.getvalue())


def add_crop_marks(pdf_buffer, mark_len=1000, mark_thickness=0.5, margin_in=0.25, gap_in=0.25):
    """
    Adds outward-facing L-shaped crop marks around each badge on a 2-up LANDSCAPE page.
    Marks begin at each badge corner and extend outward toward the page edges.
    """
    input_pdf = PdfReader(pdf_buffer)
    output_pdf = PdfWriter()

    page_width, page_height = landscape(letter)
    # Badge and page geometry (in points)
    badge_width = 4 * inch
    badge_height = 6 * inch
    gap = 0.25 * inch
    page_width, page_height = landscape(letter)

    # Center horizontally and vertically
    horizontal_margin = (page_width - (2 * badge_width + gap)) / 2
    vertical_margin = (page_height - badge_height) / 2

    # Badge coordinates
    left_x1 = horizontal_margin
    left_x2 = left_x1 + badge_width
    right_x1 = left_x2 + gap
    right_x2 = right_x1 + badge_width
    y_bottom = vertical_margin
    y_top = y_bottom + badge_height

    for page in input_pdf.pages:
        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=(page_width, page_height))
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(mark_thickness)

        def draw_outward_marks(x_left, x_right):
            """Draw marks extending OUT sfrom the badge corners"""
            offset1 = 12 * cm 
            offset2 = 1.5 * cm
            # bottom-left (outward: down & left)
            c.line(x_left - offset1, y_bottom, x_left - offset1 - mark_len, y_bottom)
            c.line(x_left, y_bottom - offset2, x_left, y_bottom - offset2 - mark_len)
            # bottom-right (outward: down & right)
            c.line(x_right + offset1, y_bottom, x_right + offset1 + mark_len, y_bottom)
            c.line(x_right, y_bottom - offset2, x_right, y_bottom - offset2 - mark_len)
            # top-left (outward: up & left)
            c.line(x_left - offset1, y_top, x_left - offset1 - mark_len, y_top)
            c.line(x_left, y_top + offset2, x_left, y_top + offset2 + mark_len)
            # top-right (outward: up & right)
            c.line(x_right + offset1, y_top, x_right + offset1 + mark_len, y_top)
            c.line(x_right, y_top + offset2, x_right, y_top + offset2 + mark_len)

        # marks for each badge
        draw_outward_marks(left_x1, left_x2)
        draw_outward_marks(right_x1, right_x2)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)
        page.merge_page(overlay_pdf.pages[0])
        output_pdf.add_page(page)

    output_buffer = BytesIO()
    output_pdf.write(output_buffer)
    output_buffer.seek(0)
    return output_buffer