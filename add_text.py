import io
import os

import fpdf
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def add_text_to_image(
    image,
    text,
    offset=(0, 0),
    font_path=None,
    font_size=200,
    text_color=(255, 0, 0),
    save=False,
):
    """
    Adds text to an image and returns the modified image in memory.

    :param image: PIL Image object.
    :param text: Text to add to the image.
    :param offset: Tuple (x, y) indicating text offset % wrt to the center.
    Center of coordinates located in the center (x positive right, y positive up)
    :param font_path: Path to a .ttf font file (default is None, which uses a system default font).
    :param font_size: Font size.
    :param text_color: Tuple (R, G, B) representing text color.
    """
    draw = ImageDraw.Draw(image)
    # Get image size
    width, height = image.size

    offset_x = offset[0] * width / 100
    offset_y = offset[1] * height / 100

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        # In case the font file is not found, use arial
        font = ImageFont.truetype("arial.ttf", font_size)

    # Get text size to center it using textbbox
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Calculate position to center text
    x = (width - text_width) // 2 + offset_x
    y = (height - text_height) // 2 - offset_y

    # Add text to image
    draw.text((x, y), text, fill=text_color, font=font)

    if save:
        output_path = "output.jpg"
        image.save(output_path)

    return image


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


def images_to_pdf(images, rows, columns, dpi=300):
    """
    Combines multiple in-memory images into a single PDF file with letter size,
    placing them on a grid of up to 6 images per page while preserving aspect ratio,
    at the specified DPI.

    :param images: List of PIL Image objects. The images to be combined into the PDF.
    :param rows: int. The number of rows to place the images on each page.
    :param columns: int. The number of columns to place the images on each page.
    :param dpi: int, optional. The DPI for the output PDF. Default is 300.
    """
    if not images:
        print("No images to convert to PDF.")
        return

    # Set page size at the desired DPI (150 DPI here)
    page_size = (612 * dpi // 72, 792 * dpi // 72)  # Convert from 72 DPI to desired DPI
    images_per_row = columns
    images_per_column = rows
    images_per_page = images_per_row * images_per_column
    max_width = page_size[0] // images_per_row - 20  # 20 for margin between images
    max_height = page_size[1] // images_per_column - 20  # 20 for margin between images

    x_offset = 10  # Initial horizontal offset
    y_offset = 50  # Initial vertical offset

    pdf_pages = []  # List to store each page as an Image object

    j = 0
    page = Image.new("RGB", page_size, "white")  # Create a new blank page
    pdf = fpdf.FPDF()
    pdf.add_page()

    for i, img in enumerate(images):
        # Resize image to fit within the max width and height while maintaining aspect ratio
        img.thumbnail((max_width, max_height))

        # Paste the image onto the page at the correct position
        page.paste(img, (x_offset, y_offset))
        pdf.image(page)

        # Update x_offset and y_offset for the next image
        if (j + 1) % images_per_row == 0:
            # Move to the next row
            x_offset = 10
            y_offset += max_height + 10  # Add some space between rows
        else:
            # Move to the next column
            x_offset += max_width + 10  # Add some space between columns

        # If we've placed 6 images, finish this page and start a new one
        if (j + 1) % images_per_page == 0:
            pdf_pages.append(page)  # Add the page to the PDF list
            page = Image.new(
                "RGB", page_size, "white"
            )  # Create a new blank page for the next set of images
            x_offset = 10  # Reset horizontal position for the new page
            y_offset = 50  # Reset vertical position for the new page
            j = 0  # Reset image count for the new page

        else:
            j += 1

    # Add any remaining images on the last page
    if j > 0:
        pdf_pages.append(page)

    # Save the images as a PDF

    # if we dump to a file directly
    # pdf_file = "test.pdf"
    # pdf_pages[0].save(pdf_file, "PDF", save_all=True, append_images=pdf_pages[1:])

    # if we do as a stream
    pdf_stream = io.BytesIO()
    pdf_pages[0].save(pdf_stream, "PDF", save_all=True, append_images=pdf_pages[1:])
    # save the PDF to a file
    # with open("outpudfsdft.pdf", "wb") as f:
    #     f.write(pdf_stream.getvalue())

    return pdf_stream


def create_pdf_name_affiliation(
    base_image, df_formatting, df_names_affiliations, df_to_pdf_options
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
        name = row.Name
        affiliation = row.Affiliation
        modified_image = apply_name_affiliation_image(
            base_image, df_formatting, name=name, affiliation=affiliation
        )
        images.append(modified_image)

    # Extract rows and columns for the PDF layout
    rows = df_to_pdf_options.loc["Number", "Rows"]
    columns = df_to_pdf_options.loc["Number", "Columns"]

    # Convert the list of images into a PDF buffer
    pdf_buffer = images_to_pdf(images, rows=rows, columns=columns)
    return pdf_buffer


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
        "card_example.jpg",
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
            "Affiliation": ["uiu", "df", "dfdf"],
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
        base_image, df_formatting, df_names_affiliations, df_to_pdf_options
    )

    # Optionally, save the PDF to a file
    with open("output_test.pdf", "wb") as f:
        f.write(pdf_buffer.getvalue())
