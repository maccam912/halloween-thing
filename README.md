# Halloween Costume Commentator 🎃

A fun Halloween project that detects people in costumes through your webcam and provides witty, AI-generated comments through a spooky voice! Perfect for Halloween parties or trick-or-treat setups.

## AI Attribution

This project was primarily developed using AI tools:
- The entire codebase was written with assistance from the "cline" VS Code extension
- Project development was guided by Claude 3.5 Sonnet
- The project itself uses AI models (YOLO, GPT-4 Vision, ElevenLabs) for its core functionality

## How It Works

1. Uses your webcam to detect people using YOLO object detection
2. When someone is detected, it captures their image
3. Sends the image to OpenAI's Vision API to analyze the costume
4. Generates a witty, contextual comment about the costume
5. Converts the comment to speech using ElevenLabs' spooky voice
6. Plays the audio through your speakers

## Requirements

- Webcam
- Speakers or audio output device
- OpenAI API key
- ElevenLabs API key
- [Pixi](https://prefix.dev/) package manager

## Hardware Setup

1. Position your webcam where it can see people in their costumes
2. Ensure your speakers are connected and at an appropriate volume
3. Make sure there's good lighting for the webcam to detect people

## Installation

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd halloween-thing
   ```

2. Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_key_here
   ```

3. Install dependencies and run the application using Pixi:
   ```bash
   pixi run python app.py
   ```

## Usage

1. Once running, the application will continuously monitor the webcam feed for people
2. When someone is detected, it will:
   - Analyze their costume
   - Generate a witty comment
   - Play the comment through your speakers
3. There's a 15-second cooldown between detections to prevent overlap
4. Press 'q' to quit the application

## Features

- Real-time person detection using YOLO
- Costume analysis using OpenAI's GPT-4 Vision
- Spooky voice generation using ElevenLabs
- Automatic audio playback
- Cool-down period to prevent spam
- Visual feedback with detection boxes

## Dependencies

- Python 3.10
- OpenAI API
- ElevenLabs API
- PyTorch with CUDA support
- Ultralytics YOLO
- OpenCV
- Pygame for audio playback
- Python-dotenv for environment variables

## Notes

- The application uses a witch voice by default (can be modified to use different voices)
- Comments are kept family-friendly for children but can be playfully witty for adults
- Ensure good lighting for better detection results
- Keep the webcam stable and at a good angle for optimal detection

## Troubleshooting

- If the webcam doesn't open, check if it's being used by another application
- If no audio plays, verify your speaker settings and connections
- If detection seems slow, ensure you have CUDA support properly configured
- If API calls fail, verify your API keys in the .env file are correct and valid
