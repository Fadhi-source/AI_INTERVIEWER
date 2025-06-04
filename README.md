🤖 Cloud AI Interviewer
Cloud AI Interviewer is an interactive web application that allows users to simulate HR and technical interviews based on their uploaded resumes. It uses the Hugging Face API for AI-powered question generation and realistic text-to-speech responses to provide a professional mock interview experience.

🚀 Features
Resume Parsing: Extracts content from uploaded PDF resumes using pdfplumber.

AI-Powered Questions: Generates relevant HR or technical questions using Mistral-7B via Hugging Face Inference API.

Voice Interaction: Converts AI questions to speech using:

Hugging Face TTS

Google Cloud TTS

Local fallback (pyttsx3)

Chat Memory: Maintains conversation flow with persistent chat history.

Gradio Interface: Easy-to-use, browser-based interface with chatbot and audio support.

📦 Tech Stack
Python 3.10+

Gradio

pdfplumber

Hugging Face Inference API

Google Cloud TTS API (optional)

pyttsx3 (optional for local TTS)

✅ Use Cases
Interview preparation for job seekers

AI-based career counseling tools

Resume-based skill evaluation platforms

🛡️ Disclaimer
This tool is for educational and personal use only. Ensure compliance with data privacy when uploading personal documents like resumes.
