# AI Models Interface

This repository contains a unified Streamlit-based web application (`app.py`) that acts as a front-end interface for interacting with various AI models. 

## 🤖 Integrated Models

Currently, the interface supports the following models:
*   **MLM (Medium Language Model):** A standard language model for general chat and text generation.
*   **MoE (Mixture of Experts):** A highly efficient model architecture that routes inputs to specialized "expert" sub-networks.
*   **SLM (Small Language Model):** A lightweight language model optimized for speed and lower resource consumption.
*   **VLM (Vision Language Model):** A multimodal model capable of understanding both text and images.
*   **SAM (Segment Anything Model):** A powerful computer vision model designed to generate high-quality object segmentation masks from images.

## 🚀 How to Start the Project (From GitHub)

If you are viewing this repository on GitHub and want to run the project locally, follow these steps in order:

### Prerequisites
1.  **Python 3.8+** must be installed on your computer.
2.  You need a valid **Groq API Key**. You can get one for free at [console.groq.com](https://console.groq.com/).

### Step-by-Step Installation

**1. Open your terminal or command prompt.**

**2. Clone the repository to your local machine:**
```bash
git clone https://github.com/jagdees2004/models.git
cd models
```

**3. Create a virtual environment (Recommended):**
This keeps the project's dependencies separate from your system Python.
```bash
python -m venv venv
```
Activate the virtual environment:
*   **Windows:** `venv\Scripts\activate`
*   **Mac/Linux:** `source venv/bin/activate`

**4. Install project dependencies:**
This will download all the required libraries to run the application.
```bash
pip install -r requirements.txt
```
*(Note: If you plan to use specific models like MoE or SAM, you may also need to install their specific requirements located in their folders, e.g., `pip install -r MoE/requirements.txt`)*

**5. Configure your API Key:**
The application needs your Groq API key to function.
*   Create a new file named exactly `.env` in the root `models` directory.
*   Open the `.env` file in any text editor and paste your key like this:
```env
GROQ_API_KEY="your_actual_api_key_here"
```
*(Note: The `.env` file is ignored by Git, so your key will remain secure on your computer.)*

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
