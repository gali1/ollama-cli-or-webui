# Project Setup and Launch Guide

## Overview

This project sets up a Flask-based web service and NON-Flask-based that can generate text responses using either the Transformers library with Hugging Face models or Llama (via `llama_cpp`). The service can be run directly or accessed via API requests.

## Prerequisites

1. **Python 3.7 or higher**: Ensure that you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

2. **Virtual Environment (recommended)**: It is good practice to create a virtual environment to manage dependencies.

3. **API Keys/Access**: Ensure you have the necessary API keys or access tokens for the Llama model if you are using it.

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies

Create a `requirements.txt` file or use the following commands to install the necessary packages:

```bash
pip install flask python-dotenv transformers torch llama_cpp
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory and set the required environment variables. Example `.env` file:

```
MODEL_NAME=gpt2
LLAMA_MODEL_PATH=/path/to/llama/model
MAX_TOKENS=256
TEMPERATURE=0.7
TOP_P=0.95
TOP_K=50
REPETITION_PENALTY=1.2
NO_REPEAT_NGRAM_SIZE=4
NUM_BEAMS=1
BATCH_SIZE=128
```

**Note:** Adjust `LLAMA_MODEL_PATH` to point to the path of your Llama model. If using the Transformers model, `LLAMA_MODEL_PATH` is not needed.

### 5. Ensure Model and Tokenizer Files

- For **Transformers**: Ensure you have internet access or the models are available locally.
- For **Llama**: Ensure the model file is available at the specified path.

## Running the Application

### 1. Launch the Flask Application

You can start the Flask server using the following command:

```bash
python <script-name>.py
```

Replace `<script-name>` with the actual name of your script file.
# llama-direct-llm-c.py
# or
# llama-direct-llm.py

### 2. Access the Application NON-FLASK APP [CLI APPROACH]:
# llama-direct-llm-c.py

```bash
> ...*prompt goes here*
```

**Example Request:**

```bash
> provide java code that generates 10 random numbers.
```

**Response:**

```json
{

import java.util.Random;

public class Main {
    public static void main(String[] args) {

        Random rand = new Random();
        for (int i = 0; i < 10; i++) {
            int randomNum = rand.nextInt((100 - 1) + 1) + 1;
            System.out.println("Random number " + (i + 1) + " : " + randomNum);
        }

    }
}


//Here we are creating a java program that generates 10 random numbers between 1 and 100. The "rand.nextInt((100 - 1) + 1) + 1" part of the code is used to generate the random numbers. The numbers generated are from the range 1 to 100(inclusive).
//"nextInt((100 - 1) + 1)" generates a random number between 0(inclusive) to 100(inclusive). Adding 1 to the end shifts that range up by 1 to make it 1 to 100.

}
```

### 3. Command-Line Interface (INFO)


You can run the script directly. You will be prompted to enter text, and you can type 'exit' to quit the interface.

### 2.5. Access the Application [Web Service APPROACH]
# llama-direct-llm.py

By default, the Flask app will run on `http://127.0.0.1:9898`. You can send a POST request to the `/generate` endpoint with a JSON payload containing the prompt:

**Example Request:**

```bash
curl -X POST http://127.0.0.1:9898/generate -H "Content-Type: application/json" -d '{"prompt": "Tell me a joke."}'
```

**Response:**

```json
{
  "response": "Here is a joke: Why did the scarecrow win an award? Because he was outstanding in his field!"
}
```

## Troubleshooting

- **Missing Environment Variables**: Ensure all required environment variables are set in the `.env` file.
- **Model Loading Issues**: Verify that model paths and configurations are correct.
- **Dependency Issues**: Ensure all dependencies are installed and compatible with your Python version.

---

Feel free to reach out if you have any questions or run into issues. Happy coding!
