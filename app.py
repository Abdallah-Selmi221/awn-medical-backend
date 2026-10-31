import gradio as gr
from main import app as fastapi_app

# Dummy UI to trick the Hugging Face Gradio SDK
demo = gr.Interface(
    fn=lambda: "Awn Medical AI - System is Active and Routing API traffic.", 
    inputs=None, 
    outputs="text"
)

# Mount the FastAPI backend underneath the Gradio app
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
