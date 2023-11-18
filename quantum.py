

'''import all the necessary libraries'''
from langchain import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from config import Settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from qiskit import *
# from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.model_selection import train_test_split


from qiskit.circuit.library import PauliFeature, AerPauliExpectation, PauliExpectation, CircuitSampler, Gradient
from qiskit.opflow import PauliFeature, AerPauliExpectation, PauliExpectation, CircuitSampler, Gradient
from qiskit_machine_learning.algorithms import QSVC
from qiskit.utils import QuantumInstance
# from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import Aer
from qiskit.opflow import CircuitSampler
from qiskit.utils import QuantumInstance
import qiskit
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from qiskit_machine_learning.algorithms import QSVC


from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import QSVC


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')


def perform_mapping(text):
        """
        Generates a mapping cluster for the provided text using the Large Language Model (LLM).

        Parameters:
            llm (LLM): An instance of the Large Language Model (LLM) for generating responses.
            text (str): The text that needs to be summarized.

        Returns:
            str: A cluster map for each text
        """

        # Create the template
        settings = Settings()

        template_string = '''categorize the following text if it is 'Science' or 'Politics' or 'Information Technology' or 'Economics' or 'Culture' or 'Entertainment' or 'Media' or 'sports'):

        Input Text:
        "{input_text}"

        Prompt Template:
        "Please categorize the given text according to one of the following four categories ('Science' , 'Politics' ,'Information Technology','Economics' , Culture')"

        Summary:
        '''
        llm = OpenAI(max_tokens=-1,  openai_api_key=settings.GPT_API_KEY)
        # LLM call
        prompt_template = ChatPromptTemplate.from_template(template_string)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run({"input_text" : text})

        return response
    



def reformatted_extracted_entities_quantum(original_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the original dataframe of entities into a melted format.

    Args:
        original_dataframe (pd.DataFrame): The original dataframe containing entities.

    Returns:
        pd.DataFrame: A restructured DataFrame with 'Entity Type' and 'Entity Value' columns.
    """

    # Use pandas' melt function to restructure the dataframe
    melted_df = original_dataframe.melt(
        var_name='Entity Type', value_name='Entity Value')

    # Drop any rows that have NaN values in 'Entity Value'
    melted_df.dropna(subset=['Entity Value'], inplace=True)

    # Drop duplicate rows based on both 'Entity Type' and 'Entity Value'
    melted_df.drop_duplicates(
        subset=['Entity Type', 'Entity Value'], inplace=True)

    # Reset the index for aesthetics
    melted_df.reset_index(drop=True, inplace=True)
    melted_df.index = np.arange(1, len(melted_df) + 1)

    return melted_df

def QML_Classification (original_dataframe: pd.DataFrame) -> pd.DataFrame:

    # Preprocessing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenization, cleaning, stopword removal, and lemmatization
    original_dataframe['processed_text'] = original_dataframe['Entity Value'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x) if word.isalnum() and word.lower() not in stop_words]))

    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(original_dataframe['processed_text'])
    Y = original_dataframe['Entity Type']

    # Splitting into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # number of steps performed during the training procedure
    tau = 100
    # regularization parameter
    C = 1000

    # Load the dataset
    feature_dim = X_train.shape[1]

    # Set up the quantum feature map
    feature_map = ZFeatureMap(feature_dimension=X_train.shape[1], reps=1)

    # Transform the dataset into quantum data
    quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))

    q_instance = QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1024)
    circuit_sampler = CircuitSampler(quantum_instance)

    # Set up the quantum kernel
    # quantum_kernel = QuantumKernel(feature_map, pauli_expansion=2, quantum_instance=q_instance)
    qkernel = FidelityQuantumKernel(feature_map=feature_map)

    # Build the QSVC model
    qsvc = QSVC(quantum_kernel=qkernel, C=C)

    # Fit the QSVC model to the training data
    qsvc.fit( X_train.toarray(), Y_train.values)

    # Predict on the test data
    y_pred = qsvc.predict(X_test.toarray())
    print("*" * 50)
    print("\n Predicted Labels:", y_pred," \n\n")

    # Apply the reformatted_extracted_entities_quantum function
    df = reformatted_extracted_entities_quantum(quantum_df)
    # Adjust the index to start from 1
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1


   # QML_Classification(df)

    return df
    




 

