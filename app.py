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
import threading
from queue import Queue
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ElevenLabs
eleven_labs = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

# TTS provider flag
USE_OPENAI_TTS = os.getenv('USE_OPENAI_TTS', 'false').lower() == 'true'

# Initialize face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise RuntimeError("Error loading face cascade classifier")

def blur_face(image, face_rect, factor=8, iterations=2):
    """Apply strong Gaussian blur to detected faces
    
    Parameters:
    - factor: Smaller values create stronger blur (default: 8)
    - iterations: Number of times to apply the blur (default: 2)
    """
    (x, y, w, h) = face_rect
    face_roi = image[y:y+h, x:x+w]
    
    # Calculate blur kernel size based on face size
    kernel_width = w // factor
    kernel_height = h // factor
    
    # Ensure kernel size is odd and at least 3
    kernel_width = max(3, kernel_width if kernel_width % 2 == 1 else kernel_width + 1)
    kernel_height = max(3, kernel_height if kernel_height % 2 == 1 else kernel_height + 1)
    
    # Apply blur multiple times for stronger effect
    blurred_face = face_roi.copy()
    for _ in range(iterations):
        blurred_face = cv2.GaussianBlur(blurred_face, (kernel_width, kernel_height), 0)
    
    image[y:y+h, x:x+w] = blurred_face
    return image

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
                        "text": "Look at this person's Halloween costume and create a witty one-liner comment. Focus only on their facial expression, what they're wearing, items they're holding, or pose they're striking. If it's a young child, be friendly and encouraging - directly compliment their specific costume choice and what makes it cool. For teens or adults, you can be more playfully sarcastic with some light roasting, using humor to keep it fun. Keep it to one natural-sounding sentence, as if someone is casually commenting while passing by."
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

def process_detection(frame, task_queue):
    """Process a detected person in a separate thread"""
    try:
        # Convert frame to base64
        image_base64 = encode_image_to_base64(frame)
        
        # Get witty comment from OpenAI
        comment = analyze_costume(image_base64)
        if comment:
            print(f"Generated comment: {comment}")
            
            # Generate and play audio
            audio_data = generate_speech(comment)
            task_queue.put(('play_audio', audio_data))
    except Exception as e:
        print(f"Error processing detection: {e}")

def audio_player_thread(task_queue):
    """Thread to handle audio playback"""
    while True:
        task = task_queue.get()
        if task[0] == 'play_audio':
            play_audio(task[1])
        elif task[0] == 'stop':
            break
        task_queue.task_done()

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
    person_detected_time = 0  # Time when a person was first detected
    frame_to_process = None  # Frame to be processed after delay
    
    # Create a queue for audio tasks
    task_queue = Queue()
    
    # Start audio player thread
    audio_thread = threading.Thread(target=audio_player_thread, args=(task_queue,), daemon=True)
    audio_thread.start()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection on original frame
            results = model(frame, verbose=False)
            
            current_time = time.time()
            person_detected = False
            
            for result in results:
                # Check if any detected object is a person (class 0)
                if any(int(box.cls) == 0 for box in result.boxes):
                    person_detected = True
                    break
            
            # Handle person detection timing
            if person_detected:
                if person_detected_time == 0:  # First detection
                    person_detected_time = current_time
                    frame_to_process = frame.copy()  # Store original frame for processing
            else:
                person_detected_time = 0  # Reset if person leaves frame
            
            # Process after 2-second delay and cooldown period
            if (person_detected_time > 0 and 
                current_time - person_detected_time >= 2 and  # 2-second delay
                current_time - last_detection_time >= cooldown_period and 
                not processing and 
                frame_to_process is not None):
                
                processing = True
                print("Processing detection after delay...")
                
                # Start processing in a separate thread with original frame
                process_thread = threading.Thread(
                    target=process_detection,
                    args=(frame_to_process, task_queue),
                    daemon=True
                )
                process_thread.start()
                
                last_detection_time = current_time
                person_detected_time = 0
                frame_to_process = None
                processing = False

            # Create display frame with blurred faces
            display_frame = frame.copy()
            
            # Detect and blur faces only for display
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                display_frame = blur_face(display_frame, (x, y, w, h))

            # Display the frame with detection boxes and blurred faces
            for result in results:
                annotated_frame = result.plot(img=display_frame)
                cv2.imshow('Halloween Costume Detector', annotated_frame)
            
            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping the application...")
    
    finally:
        # Stop audio player thread
        task_queue.put(('stop', None))
        audio_thread.join(timeout=1)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
