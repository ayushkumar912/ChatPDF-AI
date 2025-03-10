# ChatPDF AI

## Introduction
------------
ChatPDF AI is an advanced Python-based application designed to facilitate interactive engagement with multiple PDF documents using natural language processing. By integrating state-of-the-art language models, the system efficiently processes uploaded PDFs, enabling users to pose queries and receive contextually relevant responses grounded strictly in the document content. This ensures an accurate and reliable information retrieval experience.

## System Workflow
------------

![ChatPDF AI Diagram](./docs/PDF-LangChain.jpg)

The application operates through a structured sequence of processes:

1. **PDF Ingestion**: Extracts textual data from multiple uploaded PDF documents.
2. **Text Segmentation**: Divides extracted content into smaller, logically structured segments for efficient processing.
3. **Vector Representation**: Computes embeddings of segmented text to facilitate semantic search and retrieval.
4. **Query Similarity Analysis**: Matches user queries against the embedded text segments to identify the most pertinent content.
5. **Contextual Response Generation**: Utilizes the language model to synthesize responses based on the identified relevant sections.

## Installation and Configuration
----------------------------
To set up ChatPDF AI, follow these steps:

1. Clone the repository onto your local system.
2. Install the necessary dependencies via:
   ```bash
   pip install -r requirements.txt
   ```  
3. Secure an API key from OpenAI and store it in a `.env` file within the project directory:
   ```bash
   OPENAI_API_KEY=your_secret_api_key
   ``` 
   ```bash
   HUGGINGFACEHUB_API_TOKEN=your_secret_api_key
   ```  
    


## Execution and Usage
-----
To initiate and utilize ChatPDF AI:

1. Ensure all dependencies are properly installed and the API key is correctly configured.
2. Launch the application using Streamlit:
   ```bash
   streamlit run app.py
   ```  
3. The user interface will open in your web browser.
4. Upload multiple PDFs and interact with the system through the chat interface to retrieve information.

## Contribution and Licensing
------------
This project is intended for academic and research purposes, providing an open platform for personal enhancement and exploration.

ChatPDF AI is distributed under the [MIT License](https://opensource.org/licenses/MIT).

```bash
OPENAI_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
```

