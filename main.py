import gradio as gr
from groq import Groq
import base64
from PIL import Image
import io

client = Groq()

def encode_image(image, max_size=(1024, 1024)):
    # Convert numpy array to PIL Image
    img = Image.fromarray(image)
    
    # Resize image if it's larger than max_size
    img.thumbnail(max_size, Image.LANCZOS)
    
    # Convert to JPEG format
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def analyze_image(image, question):
    base64_image = encode_image(image)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llava-v1.5-7b-4096-preview",
    )
    
    # Modify this part to return a string instead of a list
    return chat_completion.choices[0].message.content

# Update the Gradio interface
demo = gr.Interface(
    fn=analyze_image,
    inputs=[
        gr.Image(type="numpy"),
        gr.Textbox(lines=2, placeholder="Ask a question about the image...")
    ],
    outputs=gr.Textbox(label="Answer", lines=10),
    title="LLaVA v1.5 7b on Groq",
    description="Upload an image and ask a question about it."
)

# Launch the app
demo.launch()