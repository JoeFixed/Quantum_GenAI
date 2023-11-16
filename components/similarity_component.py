import streamlit as st
import pandas as pd
import nltk
import spacy
nltk.download('punkt')
# Download necessary resources for Spacy
nlp = spacy.load('en_core_web_sm')
def show_similarity_component(df):
    st.title("Similarity Matcher")
    st.subheader("Most Similar Documents")

    Most_sim_num = st.slider('Select the number of most similar documents ...', 0, 10, 5)

    # Perform similarity calculations and get the most similar documents
    # You may need to add the similarity calculation logic here based on your requirements
    # The result should be a DataFrame with columns like 'Document_translated' and 'similarity_Ratio'

    # Example: Replace this with your actual similarity calculation logic
    most_similar_documents = df.nlargest(Most_sim_num, 'similarity_Ratio')

    st.write(most_similar_documents)

    with st.expander("See explanation"):
        st.write("The table above shows the most similar documents to the document you've just uploaded.")


def sentences_similartity(doc1, doc2, translated_doc):
    """
    Calculate the similarity between two columns of text in a DataFrame and store the results.
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the text data.
        doc1 (str): The name of the column representing the first document to compare.
        doc2 (str): The name of the column representing the second document to compare.
        translated_doc (str): The name of the column containing translated text for comparison.
    Returns:
        pandas.DataFrame: A new DataFrame with additional columns storing the similarity ratio between doc1 and doc2.
    Example:
        df_docs = pd.read_csv('./forigen_aff_data/Affairs_output2.csv')
        result_df = sentences_similarity(df_docs, 'doc1', 'doc2', 'new_doc')
        print(result_df)
    """
    df = pd.read_csv('./output.csv')
    result = []

    df['new_doc'] = translated_doc
    for i in df.index:
        similarity_Ratio = nlp(df[doc1][i]).similarity(nlp(df[doc2][i]))
        result.append(similarity_Ratio)
    df['similarity_Ratio'] = result
    return (df)
