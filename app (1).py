import torch
# Check if GPU (T4) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import fitz 
import docx 

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Load tokenizer and model
model_name = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

# Sumamry for single input text from user .
def summarize_text2(text, num_words=None, num_sentences=None):
    # Default lengths for sumamry
    min_len = 100
    max_len = 500

    # Adjusting max_length dynamically based on user preference
    if num_words:
        max_len = min(512, int(num_words * 1.5))
    elif num_sentences:
        max_len = min(512, int(num_sentences * 20))

    inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(model.device)
    summary_ids = model.generate(
        **inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=1.0,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Sumamry for multiple input ( Batch ) text from user .
def summarize_batch(text_list, num_words=None, num_sentences=None):
    summaries = []
    for i, text in enumerate(text_list):
        try:
            summary = summarize_text2(text, num_words=num_words, num_sentences=num_sentences)
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing document {i+1}: {e}")
            summaries.append("ERROR")
    return summaries

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

api_key = os.environ.get("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_output_tokens=1024,
    api_key = api_key
)
summary_prompt = PromptTemplate.from_template("""
You are a professional text summarizer.

Summarize the following text in a concise, coherent, and human-like manner.
Try to preserve the core message and tone.

Text:
{text}

{length_control}

Summary:
""")

gemini_chain = LLMChain(llm=llm, prompt=summary_prompt)

def summarize_with_gemini(text, num_words=None, num_sentences=None):
    length_control = ""
    if num_words:
        length_control = f"Limit the summary to around {num_words} words."
    elif num_sentences:
        length_control = f"Limit the summary to around {num_sentences} sentences."

    response = gemini_chain.invoke({
        "text": text,
        "length_control": length_control
    })

    return response['text'].strip()

import evaluate
rouge = evaluate.load("rouge")

def evaluate_summary(predicted, reference):
    scores = rouge.compute(predictions=[predicted], references=[reference])
    return scores

def evaluate_batch(predicted_list: list, reference_list: list):
    assert len(predicted_list) == len(reference_list), "Mismatch in number of predictions and references"
    scores = rouge.compute(predictions=predicted_list, references=reference_list)
    return scores


def compare_summaries(text, num_words=None, num_sentences=None):
    print("🔄 Generating PEGASUS summary...")
    summary_pegasus = summarize_text2(text, num_words=num_words, num_sentences=num_sentences)

    print("🤖 Generating Gemini summary...")
    summary_gemini = summarize_with_gemini(text, num_words=num_words, num_sentences=num_sentences)

    print("📊 Evaluating ROUGE scores...")
    rouge_scores = evaluate_summary(summary_pegasus, summary_gemini)

    return {
        "pegasus_summary": summary_pegasus,
        "gemini_summary": summary_gemini,
        "rouge_scores": rouge_scores
    }
def extract_text_from_file(file):
    file_name = file.name
    ext = file_name.split(".")[-1].lower()

    if ext == "pdf":
        text = ""
        with fitz.open(file) as doc:  # ← no file.read()
            for page in doc:
                text += page.get_text()
        return text

    elif ext == "txt":
        file.seek(0)  # ← ensure cursor is at start
        return file.read().decode("utf-8")

    elif ext == "docx":
        return "\n".join([para.text for para in docx.Document(file).paragraphs])

    else:
        raise ValueError("Unsupported file format. Please upload a PDF, TXT, or DOCX.")

import gradio as gr

# UI logic only — uses your already defined functions
def format_rouge_scores(rouge_dict):
    return "\n".join([f"{key.upper()}: {round(value * 100, 2)}%" for key, value in rouge_dict.items()])

def handle_input(file, text_input, num_words, num_sentences):
    if file is not None:
        text = extract_text_from_file(file)
    elif text_input:
        text = text_input
    else:
        return "No input provided.", "", ""

    result = compare_summaries(text, num_words, num_sentences)

    clean_pegasus = result["pegasus_summary"].replace("<n>", "\n")
    formatted_rouge = format_rouge_scores(result["rouge_scores"])

    return clean_pegasus, result["gemini_summary"], formatted_rouge

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Text Summarization: PEGASUS vs Gemini + ROUGE Evaluation")

    with gr.Row():
        file_input = gr.File(label="📤 Upload .pdf / .txt / .docx", file_types=[".pdf", ".txt", ".docx"])
        text_input = gr.Textbox(lines=6, label="✍️ Or paste your text here")

    with gr.Row():
        num_words = gr.Number(label="🎯 Word Count (optional)")
        num_sentences = gr.Number(label="🧾 Sentence Count (optional)")

    run_btn = gr.Button("🔍 Generate & Compare Summaries")

    with gr.Row():
        pegasus_output = gr.Textbox(label="📘 PEGASUS Summary", lines=10)
        gemini_output = gr.Textbox(label="🤖 Gemini Summary", lines=10)
        rouge_output = gr.Textbox(label="📊 ROUGE Scores", lines=10)

    run_btn.click(
        fn=handle_input,
        inputs=[file_input, text_input, num_words, num_sentences],
        outputs=[pegasus_output, gemini_output, rouge_output]
    )

demo.launch(share=True)
