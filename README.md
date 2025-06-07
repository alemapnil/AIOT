# AIOT
## Description
Welcome to my project ! This is an AI robot allowing you to realise 3 functions such as playing the sound, speech to text, detecting the number of human beings through your command of voice. This application can only be executed in the version of __Python 3.11.0__. I promise you'll find it very interesting and comprehend how formidable AI's power is. 
## Motivation
I'm curious about the achievement that AI can fulfil, and I'm also eager to apply new technologies to my project to upgrade my capacity of software engineering. 
## OpenAI
#### Audio API in Realtime API
The transcriptions API, belonging to audio API, takes as input the audio file you want to transcribe and the desired output file format for the transcription of the audio. It utilises the LLM model, GPT-4o. 
```
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
with open(audio, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model = "gpt-4o-transcribe", 
        file = audio_file
    )
print(transcription.text)
```
#### Response API
OpenAI's most advanced interface for generating model responses. Supports text and image inputs, and text outputs. Allow the model access to data using function calling. It exerts the brand-new LLM model, GPT-4.1, which was just launched in April, 2025 and outperforms GPT-4o. With this new model, OpenAI can decide which function should be called based on the content of input.
```
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": content}],
    tools = [A_tool,B_tool, C_tool]
)
print(response.output_text)
```
## Installing Environment
### Python 3.11.0
Deploying the environment with __Python 3.11.0__ so that _tensorflow_ can be installed successfully.
### requirements.txt
Creating an environment as the same as the developer.
```
pip install -r requirements. txt
```
### ffmpeg
Installing _FFMPEG_, a multimedia framework, for you to convert a file format of video and audio.
The way of its installation based on the type of OS in your computer.
### Configuration of Yolov3
Downloading the file, ***yolov3.cfg***, I attached in this repository, and [***yolov3.weights***](https://data.pjreddie.com/files/yolov3.weights). Put those two files in this directory of relative path, ***./.cvlib/object_detection/yolo/yolov3***.
_Yolov3_ is the model for detection of objects in camera, so you must set its configuration with ***yolov3.cfg*** and ***yolov3.weights*** in the right place. 
## Packages
#### playsound
Playing the sound from speakers in your computers.
#### pyaudio, pydub, boto3
Recording audio and outputing audio file with _pyaudio_.

Converting audio format with _pydub_ from .wav to .mp3 owing to smaller size of .mp3, resulting in faster transmission to AWS.

Creating the folder in AWS storage space and uploading the files to it with _boto3_.
#### openCV-python, cvlib
Opening the camera embedded in your computers and showing the video of objects in front of screen with _openCV-python_.

Detecting the number of each kind of objects in the video with _cvlib_.
## How to Use the App?
Please talk to AI your request within a short period of time when you see the below sentence.
> Talk to AI within this 7 seconds ......

AI will operate the functions I designed if you say something about playing the sound, STT or detecting the number of human beings. 
