import gradio as gr
from transformers import pipeline
import torch


translator = pipeline(task="translation",
                      model="facebook/nllb-200-distilled-600M",
                      torch_dtype=torch.bfloat16) 

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")


def translate(text, src_lang="amh_Ethi", tgt_lang="eng_Latn"):
    text_translated = translator(text,
                             src_lang=src_lang,
                             tgt_lang=tgt_lang)
    
    text = list(text_translated[0].values())[0]
    result = classification(text)
    
    return list(text_translated[0].values())[0], result

def classification(text, candidate_labels=["Very Negative", "Negative", "Neutral", "Postive", "Very Positive"]):
    try:
        output = classifier(text, candidate_labels, multi_label=False)
        labels_order = output['labels']
        scores_order = output['scores']
        results = [f"{label} has a {score*100:.2f}%" for label, score in zip(labels_order, scores_order)]
        return '\n'.join(results)
    except Exception as e:
        return str(e)




with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            seed = gr.Text(label="Input Amharic Text")
        with gr.Column():
            translated_english = gr.Text(label="Translated English Text")
            classified = gr.Text(label="Zero Shot Classification")
    btn = gr.Button("Translate and classify")
    btn.click(translate, inputs=[seed], outputs=[translated_english, classified])
    # gr.Examples(["My name is Clara and I am"], inputs=[seed])

demo.launch()
