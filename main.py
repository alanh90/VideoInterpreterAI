import streamlit as st
import cv2
import base64
import tempfile
import openai as OpenAI
import os
from dotenv import load_dotenv
import requests

load_dotenv()
client = OpenAI.Client(api_key=os.getenv("OPENAI_API_KEY"))


def main():
    st.title("Video Understanding app")

    # Video Upload Section
    uploaded_video = st.file_uploader('Upload a video', type=('mp4', 'avi', 'mov'))

    if uploaded_video is not None:
        # Save the video to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()

        # Process the video display the first image
        base64_frames = video_to_base64_frames(tfile.name)
        st.image(base64.b64decode(base64_frames[0]), caption='First frame of the video')

        # Generate description of the video
        description = generate_description(base64_frames)
        if description:
            st.write("Description", description)
        else:
            st.write("Description could not be generated.")

        # Clean up the temp file
        os.unlink(tfile.name)


def video_to_base64_frames(video_file_path):
    # Logic is going to go
    video = cv2.VideoCapture(video_file_path)
    base64_frames = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')
        base64_frames.append(base64_frame)
    video.release()
    return base64_frames


def generate_description(base64_frames):
    try:
        prompt_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate a description for this sequence of video frames."},
                    *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}", "detail": "low"}}, base64_frames[0::15]),
                ]
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=prompt_message,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except OpenAI.APIError as e:
        st.error("OpenAI error try again")
        st.write('Error details', e)
        retry_button = st.button('Retry Request')
        if retry_button:
            return generate_description(base64_frames)
        return None


if __name__ == '__main__':
    main()
