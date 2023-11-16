from PIL import Image
import pytesseract
import re


def perform_ocr(uploaded_image):
    print(f"the image uplodes path {uploaded_image}")
    custom_config = r'-l eng+ara --psm 6'
    extracted_information = pytesseract.image_to_string(uploaded_image, config=custom_config)
    arabic_text = ' '.join(re.findall(r'[\u0600-\u06FF0-9_]+', extracted_information))
   # arabic_text=pytesseract.image_to_string(uploaded_image , lang='ara',config= ".")
    return arabic_text

