import cv2
from ultralytics import YOLO
from openai import OpenAI
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import base64
import time
import pygame
from io import BytesIO
import tempfile

# Load environment variables
load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ElevenLabs
eleven_labs = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

# TTS provider flag
USE_OPENAI_TTS = os.getenv('USE_OPENAI_TTS', 'false').lower() == 'true'

def encode_image_to_base64(image):
    """Convert CV2 image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def analyze_costume(image_base64):
    """Analyze the image using OpenAI's Vision API"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Look at this Halloween costume and come up with a witty, clever one-liner comment about it. Keep it fun and lighthearted. Make it sound natural, as if someone was casually passing by. Keep it brief - one sentence only. If there are any young children in the photo, keep it friendly and compliment them, but if there are only teenagers and adults they will think some gentle ribbing is funny."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=100
    )
    return response.choices[0].message.content

def generate_speech(text):
    """Generate speech using either OpenAI or ElevenLabs"""
    if USE_OPENAI_TTS:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        return response.content
    else:
        # Generate audio using ElevenLabs
        audio = eleven_labs.text_to_speech.convert(
            text=text,
            voice_id="vGQNBgLaiM3EdZtxIiuY",  # Witch voice
            model_id="eleven_turbo_v2"
        )
        # Convert the generator to bytes
        return b''.join(chunk for chunk in audio if chunk is not None)

def play_audio(audio_data):
    """Play audio using pygame"""
    # Save audio data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        temp_file.write(audio_data)
        temp_path = temp_file.name

    # Initialize pygame mixer and play the audio
    pygame.mixer.init()
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()
    
    # Wait for the audio to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    # Clean up
    pygame.mixer.quit()
    os.unlink(temp_path)

def main():
    # Initialize YOLO
    model = YOLO("yolov8n.pt")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print(f"Using {'OpenAI' if USE_OPENAI_TTS else 'ElevenLabs'} for text-to-speech...")
    print("Waiting for people to be detected...")
    
    last_detection_time = 0
    cooldown_period = 15  # Seconds between detections
    processing = False  # Flag to track if we're currently processing a detection
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model(frame, verbose=False)
            
            current_time = time.time()
            # Only process new detections if we're not currently processing and cooldown period has passed
            if not processing and current_time - last_detection_time >= cooldown_period:
                for result in results:
                    # Check if any detected object is a person (class 0)
                    if any(int(box.cls) == 0 for box in result.boxes):
                        processing = True  # Set processing flag
                        print("Person detected! Analyzing costume...")
                        
                        try:
                            # Convert frame to base64
                            image_base64 = encode_image_to_base64(frame)
                            
                            # Get witty comment from OpenAI
                            comment = analyze_costume(image_base64)
                            if comment:  # Check if comment is not None
                                print(f"Generated comment: {comment}")
                                
                                # Generate and play audio
                                audio_data = generate_speech(comment)
                                play_audio(audio_data)
                            
                            last_detection_time = current_time
                        except Exception as e:
                            print(f"Error processing detection: {e}")
                        finally:
                            processing = False  # Reset processing flag
                        break

            # Display the frame with detection boxes
            for result in results:
                annotated_frame = result.plot()
                cv2.imshow('Halloween Costume Detector', annotated_frame)
            
            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping the application...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
