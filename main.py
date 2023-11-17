import streamlit as st
from config import Settings
from utils.ocr_utils import perform_ocr
from utils.translation_utils import perform_translation, load_translation_models
from utils.summarization_utils import SummarizationUtils
from components.similarity_component import sentences_similartity
from utils.ner_utils import (
    ner_spacy,
    load_ner_model,
    extract_entities_from_text,
    prioritize_entities,
    # QML_Classification
)
from utils.graph import create_graph_network_new_approach1
from utils.gpt import GPTFunction
from spacy_streamlit import visualize_ner
from PIL import Image
import pandas as pd
from config import Settings
import tempfile
from pdf2image import convert_from_path
import pytesseract
import torch
import toml
from docx import Document


IMAGE_EXTENSIONS = ("png", "jpg", "jpeg")
settings = Settings()

ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg", "pdf", "docx", "txt")
quantum_df = pd.DataFrame()

theme_config_path = "config.toml" 
theme_config = toml.load(theme_config_path)

  
st.markdown(f"""
    [theme]
    primaryColor="{theme_config['theme']['primaryColor']}"
    backgroundColor="{theme_config['theme']['backgroundColor']}"
    secondaryBackgroundColor="{theme_config['theme']['secondaryBackgroundColor']}"
    textColor="{theme_config['theme']['textColor']}"
    font="{theme_config['theme']['font']}"
""", unsafe_allow_html=True)

@st.cache_resource(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_models():
   return load_translation_models()


def main():
    # Load translation models
    (
        model_ar_en,
        tokenizer_ar_en,
        model_en_ar,
        tokenizer_en_ar,
    ) = load_models()
    # Center the logo at the top of the main page
    st.image(
        "imgs/Quantum_design5.png",
        width=10,
        use_column_width="always",
    )

    # Your main function here, use the utility functions
    # st.title("Quantum-GenAI")

    # st.subheader(
    #     "Data to Decisions: Scan, Recognize, Summarize. Dive into a world of smart analysis and tailored recommendations with just a click. Your journey to insights starts here!"
    # )
    # st.subheader(
    #     "Master Your Unstructured Documents"
    # )
    with st.sidebar.container():
        # Test with a different local image or an online image URL
        st.image(
            "imgs/rsz_docunify.png", width=10, use_column_width=True
        )

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        # st.subheader("Main Page")
        ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg", "pdf", "docx", "txt")
        uploaded_files = st.file_uploader(
            "Upload a file", type=ALLOWED_EXTENSIONS, accept_multiple_files=True
        )

        if uploaded_files:
            selected_file = st.selectbox(
                "Apply processing on only one of the uploaded files.", uploaded_files
            )
            file_type = selected_file.type

            if file_type == "application/pdf":
                # Create a temporary file to save the uploaded PDF
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmpfile:
                    tmpfile.write(selected_file.getvalue())

                # Convert PDF pages to images using the path of the temporary file
                images = convert_from_path(tmpfile.name)

                # Use Tesseract to OCR the images
                text_content = ""
                for image in images:
                    text_content += pytesseract.image_to_string(image, lang="ara")
                arabic_text = text_content

            # For Images
            elif file_type in ["image/png", "image/jpeg"]:
                image = Image.open(selected_file)
                st.image(image, caption="Uploaded Image.", use_column_width=True)
                arabic_text = perform_ocr(image)

            # For DOCX files
            elif (
                file_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                doc = Document(selected_file)
                fullText = []
                for para in doc.paragraphs:
                    fullText.append(para.text)
                text_content = "\n".join(fullText)
                arabic_text = text_content

            # For TXT files
            elif file_type == "text/plain":
                text_content = selected_file.read().decode()
                arabic_text = text_content

            # Translate to English
            translated_doc = perform_translation(
                arabic_text, model_ar_en, tokenizer_ar_en
            )
            rec_translated_doc = translated_doc

            # Retranslate to Arabic
            arabic_text = perform_translation(
                translated_doc, model_en_ar, tokenizer_en_ar
            )

            options = ["English Document", "Arabic Document"]
            selection = st.sidebar.selectbox(
                "Choose OCR result's Language", options, index=1
            )
            if selection == "English Document":
                st.success(translated_doc, icon="✅")
            elif selection == "Arabic Document":
                st.success(arabic_text, icon="✅")

                #############################################################
                # Function for Sumy Summarization
                # Summaryzer Streamlit App
                st.title("Document Summarizer")
                st.subheader("Summarization text of the inserted Document")
                # let's delete all the other models
                summarizer = SummarizationUtils()

                # Generate the summary
                summary_result = summarizer.perform_summarization(translated_doc)

                summary_result = perform_translation(
                    summary_result, model_en_ar, tokenizer_en_ar
                )
                st.success(summary_result)
                #############################################################
                # Save similarity output in separate container
            with st.container():
                st.title("Similarity Matcher")
                st.subheader("Most Similar Documents")
                Most_sim_num = st.slider(
                    "Select the number of most similar documents ...", 0, 10, 5
                )

                new_df = sentences_similartity(
                    "new_doc", "Document_translated", translated_doc
                )
                new_df = new_df.nlargest(Most_sim_num, ["similarity_Ratio"])[
                    ["Document_translated", "similarity_Ratio"]
                ]
                st.write(new_df)

                with st.expander("See explanation"):
                    st.write(
                        "The table above shows most similar documents to the document you've just uploaded. \n It's supposed to be discussing *one Topic* You can download it and give it a look."
                    )
                    st.image(
                        "https://cdn0.iconfinder.com/data/icons/flatie-action/24/action_005-information-detail-notification-alert-512.png",
                        width=100,
                    )

                # Extract named entities
                doc = ner_spacy(translated_doc)
                st.subheader("Named Entity Recognition")
                visualize_ner(doc)

                # NER
                # model_NER, tokenizer_NER = load_ner_model()
                df_graph = extract_entities_from_text(arabic_text)

                # Extract named entities from the 5 most similar documents and translating them to arabic
                result_df = pd.DataFrame()
                for i in new_df.index:
                    # Translate the document

                    translated_doc = new_df["Document_translated"][i] + "" + "."
                    df_graph = extract_entities_from_text(translated_doc)

                    # Concatenate the result
                    result_df = pd.concat([result_df, df_graph], ignore_index=True)

                    # prioritize_entities  will organize dataframe by the priority of the entity types
                result_df = prioritize_entities(result_df)
                st.write(result_df)
                # quantum_df = result_df

                # Display the DataFrame as a table in Streamlit
                st.subheader("Knowledge Graph")

                graph_data = pd.DataFrame()

                for i in new_df.index:
                    # Translate the document
                    translated_doc = perform_translation(
                        new_df["Document_translated"][i], model_en_ar, tokenizer_en_ar
                    )

                    arabic_text = (
                        arabic_text
                        + ""
                        + "."
                        + "next is a new document."
                        + translated_doc
                    )
                    arabic_text = summarizer.perform_summarization(arabic_text)

                physics = st.checkbox("Add Physics Interactivity", value=True)

                GraphText = arabic_text  # replace with your actual data
                create_graph_network_new_approach1(
                    GraphText, model_en_ar,tokenizer_en_ar
                )  # Assuming your new function takes a text input
                # Generating ChatGPT Report in Arabic and English
                st.subheader(
                    "Generation of an Analytical Report with Recommendations, Empowered by Generative AI, Based on the Provided Document."
                )
                prompt_topicModeling = "Please provide a comprehensive analysis of the following text, taking into consideration the current trends and developments in this area. Your analysis should be thorough and reflect deep expertise in the subject matter. After examining the text, offer specific, well-informed recommendations based on your findings and the knowledge sources at your disposal. Here is the text for your review: "
                analytics_text = GPTFunction(rec_translated_doc, prompt_topicModeling)
                st.write(analytics_text)
                translated_recommendation = summarizer.perform_summarization(
                    translated_doc
                )
                translated_recommendation = perform_translation(
                    translated_recommendation, model_en_ar, tokenizer_en_ar
                )
                st.subheader("(Generative AI)تقرير تحليلي مع توصيات  مبنية باستخدام")
                st.write(translated_recommendation)
                st.text("========\n")
              #  QML_Classification(quantum_df)

if __name__ == "__main__":
    main()
