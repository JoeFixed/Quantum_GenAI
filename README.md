


<img
  src="imgs/Quantum_design_cover"
  alt="Alt text"
  title="Quantum Gen AI"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

# Quantum Gen AI

Quantum Gen AI is a solution intially designed to process documents in image format and harness unstructured data for practical applications. Here are its key functionalities:

- It performs Optical Character Recognition (OCR) to extract text.
- It generates text summaries.
- It retrieves the most similar documents from a database based on the document's topic.
- It creates network graphs with named entity recognition.
- It provides document recommendations.


        


QuntumGenAi
└───utils # source folder
    │   ocr_utils.py
    │   translation_utils.py
    │   summarization_utils.py
    │   ner_utils.py
    │   graph_utils.py
    │   gpt_utils.py
    │   prepare_quantum.py
    │   quantum.py
└───compnents
        │   ner_component.py
        │   similarity.py    
           
└───data
      output.csv
├── main.py
├── config.py
├── requirements.txt
├── requirements_quantum.txt
└── README.md                

   
 ## Technolgies


- OpenAI
- Pytesseract 
- Langchain / LamaIndex
- Networkxx
- Quantum Machine learning
- Transormers (Hugging face)


    


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
