from flask import Flask, render_template, request
from openai import OpenAI
from dotenv import load_dotenv
import os
from moviepy import VideoFileClip
import whisper

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Create Groq client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""

    if request.method == "POST":
        transcript = request.form.get("transcript")

        if transcript.strip() == "":
            summary = "Please enter a meeting transcript."
        else:
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an AI meeting assistant. "
                                "Summarize the meeting and provide:\n"
                                "1. Summary\n"
                                "2. Key Points\n"
                                "3. Action Items"
                            ),
                        },
                        {
                            "role": "user",
                            "content": transcript,
                        },
                    ],
                    temperature=0.3,
                )

                summary = response.choices[0].message.content

            except Exception as e:
                summary = f"Error: {str(e)}"

    return render_template("index.html", summary=summary)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/process", methods=["POST"])
def process():

    if "file" not in request.files:
        return "No file selected."

    file = request.files["file"]

    if file.filename == "":
        return "No file selected."

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

   # Convert video to audio if uploaded file is MP4
    if filepath.endswith(".mp4"):
        video = VideoFileClip(filepath)
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "audio.wav")
        video.audio.write_audiofile(audio_path)
    else:
        audio_path = filepath

    # Convert speech to text
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)

    transcript = result["text"]

    # Send transcript to Groq AI
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Summarize the meeting and provide Summary, Key Points, and Action Items."
            },
            {
                "role": "user",
                "content": transcript
            }
        ]
    )

    summary = response.choices[0].message.content

    return render_template(
        "index.html",
        transcript=transcript,
        summary=summary
    )

if __name__ == "__main__":
    app.run(debug=True)