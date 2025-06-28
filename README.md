# 📄 PEGASUS + Gemini Summarizer with ROUGE Evaluation

This project is an advanced text summarization tool that compares two powerful approaches—**PEGASUS** (by Google via HuggingFace Transformers) and **Gemini 1.5 Flash** (via LangChain)—and evaluates their output using ROUGE metrics.

It offers a user-friendly Gradio interface for uploading documents or pasting text, setting summary length preferences, and comparing results side by side.

---

## 🚀 Features

- 📝 **Summarization via PEGASUS**: Using Google's pretrained `pegasus-cnn_dailymail` model.
- 🤖 **Summarization via Gemini**: Using Google's Gemini 1.5 Flash through LangChain.
- 📊 **Evaluation with ROUGE**: Uses HuggingFace's `evaluate` library to compute ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.
- 📂 **Multiformat Input**: Supports `.pdf`, `.txt`, `.docx` uploads or raw text input.
- 🎛️ **Length Controls**: Optional sliders for word count or sentence count.
- 🧠 **LangChain Integration**: Demonstrates chaining Gemini model with prompts.
- 🎨 **Clean Gradio UI**: Easy-to-use interface deployed via Hugging Face Spaces.

---

## 🛠️ Technologies Used

| Tech              | Purpose                                      |
|-------------------|----------------------------------------------|
| 🤗 Transformers   | PEGASUS model for summarization              |
| 🧱 LangChain       | Gemini 1.5 Flash LLM chaining                |
| 🌐 Gradio          | Frontend interface                          |
| 📊 evaluate        | ROUGE metric calculation                    |
| 📄 pymupdf/docx    | PDF/DOCX text extraction                    |
| ⚡ CUDA (T4)        | GPU acceleration in Google Colab & Spaces   |

---

## 📷 Demo

Deployed on **Hugging Face Spaces**  
🔗 [Live Demo](https://huggingface.co/spaces/Divyansh-87/PEGASUS_Summarizer)  

---

## 🧪 How It Works

1. **User Inputs Text** (or uploads file)
2. **PEGASUS Generates Summary**
3. **Gemini Generates Summary via LangChain**
4. **ROUGE Metrics Computed** between the two summaries
5. **All results displayed side-by-side**, with an option to download them as PDF (optional)

---

## 📁 File Structure
```bash
├── app.py # Main Gradio app
├── requirements.txt # Required dependencies
├── README.md 
```

## 
---

## ⚙️ Installation

```bash
git clone https://github.com/Divyansh8791/PEGASUS_text_summarizer.git
cd PEGASUS_text_summarizer
```
## Install dependencies
```bash
pip install -r requirements.txt
```
## Run the app
```bash
python app.py
```
---
## 👨‍💻 Author
Divyansh – AI/ML Enthusiast
✉️ [LinkedIn](https://wwww.linkedin.com/in/divyansh-dhiman)

---
