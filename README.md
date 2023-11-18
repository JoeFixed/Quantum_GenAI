# MFA

MFA is a tool designed to process documents in image format and harness unstructured data for practical applications. Here are its key functionalities:

- It performs Optical Character Recognition (OCR) to extract text.
- It generates text summaries.
- It retrieves the most similar documents from a database based on the document's topic.
- It creates network graphs with named entity recognition.
- It provides document recommendations.

    .
    ├── data                    
    │   ├── output.csv                 
    ├── imgs            
    ├── utiils         
    ├── components
         ├── ner_component  
         ├── similarity_component
    ├── main.py
    ├── config.py
    ├── requirements.txt
    ├── requirements_quantum.txt
    


## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository_URL>
   cd MFA

2. Install the required Python packages by running:
    
   ```bash
   pip install -r requirements.txt

3. Install the required Python packages by running:

   ```bash
    pip install -r requirements_quantum.txt

4. Run the application using Streamlit:

   ```bash
   streamlit run main.py
