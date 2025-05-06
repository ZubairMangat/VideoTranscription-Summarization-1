import whisper
from moviepy.editor import VideoFileClip
from transformers import BartForConditionalGeneration, BartTokenizer
import streamlit as st
import os
import tempfile
import torch

# Load models
st.title("🎥 Video Transcription and Summarization")
st.write("Upload a video to transcribe the audio and generate a summary.")

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return whisper_model, tokenizer, summarizer_model

whisper_model, tokenizer, summarizer_model = load_models()

# Upload video
video_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi", "mkv"])

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        tmp_video_path = tmp_video.name

    # Extract audio and save as temp file
    with st.spinner("Extracting audio..."):
        try:
            clip = VideoFileClip(tmp_video_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                audio_path = tmp_audio.name
                clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

            # Transcribe audio
            st.spinner("Transcribing audio...")
            result = whisper_model.transcribe(audio_path)
            transcribed_text = result['text']
            st.subheader("📜 Transcribed Text")
            st.text_area("", transcribed_text, height=200)

            # Summarize
            st.spinner("Summarizing...")
            inputs = tokenizer(transcribed_text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = summarizer_model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            st.subheader("📝 Summary")
            st.text_area("", summary, height=150)

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
