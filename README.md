# AI Models Interface

This repository contains a unified Streamlit-based web application (`app.py`) that acts as a front-end interface for interacting with various AI models. 

## 🤖 Integrated Models

Currently, the interface supports the following models:
*   **MLM (Medium Language Model):** A standard language model for general chat and text generation.
*   **MoE (Mixture of Experts):** A highly efficient model architecture that routes inputs to specialized "expert" sub-networks.
*   **SLM (Small Language Model):** A lightweight language model optimized for speed and lower resource consumption.
*   **VLM (Vision Language Model):** A multimodal model capable of understanding both text and images.
*   **SAM (Segment Anything Model):** A powerful computer vision model designed to generate high-quality object segmentation masks from images.

## 🚀 Setup & Installation

### Prerequisites
1.  Python 3.8+ (Recommended)
2.  A valid [Groq API Key](https://console.groq.com/).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jagdees2004/models.git
    cd models
    ```

2.  **Install dependencies:**
    Install the core requirements from the root directory. You may also need to install specific requirements located within individual model directories (e.g., `MoE/requirements.txt`, `SAM/requirements.txt`).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY="your_api_key_here"
    ```
    *Note: The `.env` file is included in `.gitignore` to prevent accidental exposure of your keys to GitHub.*

### Running the Application

Start the Streamlit development server:

```bash
streamlit run app.py
```

Or use the provided batch script on Windows:
```cmd
run_app.bat
```

The application will be accessible in your web browser, typically at `http://localhost:8501`.

## 📂 Project Structure

*   `app.py`: The main Streamlit application script containing the UI and routing logic.
*   `MLM/`, `MoE/`, `SLM/`, `VLM/`, `SAM/`: Directories containing the specific implementation logic for each respective model.
*   `requirements.txt`: Global dependencies for the project.
*   `.env`: (Ignored by Git) Local environment variables for storing sensitive API keys.
*   `*.pt`: (Ignored by Git) Large model weights (e.g., `FastSAM-s.pt`, `yolov8n-seg.pt`).

## 🛠️ Architecture

Instead of housing the complex model execution logic, `app.py` acts purely as an interface. It delegates all AI operations to the localized modules (`MLM.mlm`, `SAM.sam`, etc.), keeping the frontend clean and making it easy to add new models in the future.
