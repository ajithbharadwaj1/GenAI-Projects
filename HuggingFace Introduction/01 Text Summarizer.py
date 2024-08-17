from transformers import pipeline
import gradio as gr
get_completion = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']




gr.close_all()
demo = gr.Interface(fn=summarize,
                    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Result", lines=3)],
                    title="Text summarization"

                   )
demo.launch(share=True)