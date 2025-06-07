import pyaudio, time, wave, uuid, os, boto3, cv2 # openCV-python
import cvlib, traceback
from cvlib.object_detection import draw_bbox
from playsound import playsound
from pydub import AudioSegment
from opencc import OpenCC
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def voice(): ## playsound
    print('What a lovely sound ^_^')
    playsound('https://d3i2i3wop7mzlm.cloudfront.net/github/seawaves.mp3')
    return 'Feel relaxed'


def transcribe_and_s3(audio, words, bucket):  ## transcribe and upload to AWS

    ## 用OpenAI的API做STT的速度及精準度都比免費開源的whisper好太多，30秒音檔10秒內轉譯、上傳s3完畢，且文字正確
    ## see Realtime API (Audio API) https://platform.openai.com/docs/guides/speech-to-text

    with open(audio, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model = "gpt-4o-transcribe", 
            file = audio_file
        )

    # 中文簡轉繁
    converted = OpenCC('s2t').convert(transcription.text)
    print(converted)

    ### txt
    with open(words,'w',encoding='utf-8') as f:
        f.write(converted)

    # upload to AWS

    s3.upload_file(audio, bucket, f'{s3folder}/{audio}')
    s3.upload_file(words, bucket, f'{s3folder}/{words}')

    print('Audio is in your s3 >>',f'./{s3folder}/{audio}')
    print('Txt is in your s3 >>',f'./{s3folder}/{words}')
    

def SST(): ## record audio, and insert function <transcribe_and_s3>
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    SECONDS = 30

    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK)
    print('Please strat speaking for 30 seconds now ......')

    ## record frames
    frames = []
    for i in range(int(RATE/CHUNK* SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    ## stop recording
    stream.stop_stream()
    stream.close()
    p.terminate

    print('Recording stoped !')

    unique = uuid.uuid1() ## make a UUID based on the host ID and current time
    wavfile, mp3file, txtfile  = f'{unique}.wav', f"{unique}.mp3",f'{unique}.txt'

    # produce wav file
    with wave.open(wavfile,'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    # convert to mp3
    song = AudioSegment.from_wav(wavfile)      
    song.export(mp3file, format="mp3")
    # delete wav
    while os.path.isfile(mp3file) is not True:
        print('wait')

    transcribe_and_s3(mp3file,txtfile,os.getenv("AWS_BUCKET_NAME")) ## transcribe and upload to AWS

    return '語音與文字上傳完畢'


def human_watch(): ## human detection
    ## open camera
    print('Please wait a few seconds for camera, thank you.')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    print('Camera is ready to film.')
    ## filming...
    while cap.isOpened():
        ret, frame = cap.read()
        ## model yolov3-tiny is fast but not accurate
        bbox, label, conf = cvlib.detect_common_objects(frame,
                                                model ='yolov3', 
                                                confidence = 0.5,
                                                enable_gpu = False
        )

        ## draw bounding box over detected objects
        out = draw_bbox(frame, bbox, label, conf)
        ## display output
        cv2.imshow("Human being detection", out)

        person_count = len([l for l in label if l == 'person'])
        l = set(label)  # l refers to other items 
        l.remove('person') if 'person' in l else False

        print('The number of figure now is',person_count)
        
        key = cv2.waitKey(1)
        ## if user press any keys on the keyboard, Window will be closed.
        # if key > 0:
        #     break
        if cv2.getWindowProperty('Human being detection', cv2.WND_PROP_VISIBLE) < 1:
            print("ALL WINDOWS ARE CLOSED")
            break

    cap.release()
    ## release resources
    cv2.destroyAllWindows()

    return 'nice camera'


A_tool = {
    "type": "function",
    "name": "voice",
    "description": "Play sound or make some voice."
}
B_tool = {
    "type": "function",
    "name": "SST",
    "description": "Speech To Text, transcribing audio to words"
}
C_tool = {
    "type": "function",
    "name": "human_watch",
    "description": "detection of the number of human beings"

}

def call_function(name): ## execute the function that Ai decide
    if name == "voice":
        return voice()
    if name == "SST":
        return SST()
    if name == 'human_watch':
        return human_watch()



def confirm_box(bucket): # confirm an ai folder is None or not, If not, create a new one.
    global s3folder
    paginator = s3.get_paginator('list_objects_v2')
    folder_names = set()  # containing the current folders' name
    for page in paginator.paginate(Bucket=bucket, Prefix='', Delimiter='/'):
        # page is a dictionary , the key 'CommonPrefixes' contains current folder's name
        if 'CommonPrefixes' in page:
            for prefix in page['CommonPrefixes']:
                folder_names.add(prefix['Prefix'])

    # create folder for my uploaded files
    s3folder = 'ai_pam'
    if s3folder + '/' not in folder_names:
        s3.put_object(Bucket=bucket, Key= s3folder + '/')


def receive(): ## python receive audio commands from users 5 seconds a time
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    SECONDS = 7

    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK)
    print('Talk to AI within this 7 seconds ......')

    frames = []
    for i in range(int(RATE/CHUNK* SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate

    print('Stop giving command !')

    t = time.localtime()
    year, mon, day, hour, minute, sec = t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec

    cmdwv = f"{year}{mon:02d}{day:02d}_{hour:02d}{minute:02d}{sec:02d}.wav"

    # produce command's  wav file
    with wave.open(cmdwv,'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    ## request STT to ai
    ## see Realtime API (Audio API) https://platform.openai.com/docs/guides/speech-to-text

    with open(cmdwv, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model = "gpt-4o-transcribe", 
            file = audio_file
        )

    # 中文簡轉繁
    user_cmd = OpenCC('s2t').convert(transcription.text)
    print("Your command is >>",user_cmd)
    return user_cmd


## create AWS storage instance
s3 = boto3.client("s3",
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key = os.getenv("AWS_SECRET_KEY")
    )

## confirm AwS folder is None or not.
confirm_box(os.getenv("AWS_BUCKET_NAME"))

## create AI instance
client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
    )


while True:
    print('===== Asking OpenAI =====')
    try:
        content = receive()
        ## AI choose a right function to response
        ## see response API https://platform.openai.com/docs/api-reference/responses

        response = client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": content}],
            tools = [A_tool,B_tool, C_tool]
        )
        # response.output is a list, one element inside
        for tool_call in response.output:
            if tool_call.type == "function_call":
                call_function(tool_call.name) 
            elif tool_call.type == "message":
                print('>>',response.output_text)
            else:
                print('Somethong goes wrong >',tool_call)
    except KeyboardInterrupt:
        print('Bye Bye !')
        break
    except:
        print(traceback.format_exc())
    finally:
        for i in os.listdir():
            os.remove(i) if i.endswith('.wav') or i.endswith('.mp3') or i.endswith('.txt') else False




