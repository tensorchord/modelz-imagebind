from typing import List, Literal, TypedDict
import httpx
import msgpack

with open("dog_audio.wav", "rb") as f:
    dog_audio_bytes = f.read()

with open("dog_image.jpg", "rb") as f:
    dog_image_bytes = f.read()

with open("dog_video.mp4", "rb") as f:
    dog_video_bytes = f.read()

# Change to real endpoint URL and token
REMOTE_URL = "https://xxx.modelz.tech/"
TOKEN = "mzi-ttt"

class Input(TypedDict):
    task: Literal["image", "audio", "video", "text"]
    data: List[bytes | str]

cases: List[Input] = [
    {"task":"text", "data": ["A dog", "doggery", "puppy"]},
    {"task":"image", "data":[dog_image_bytes]},
    {"task":"audio", "data":[dog_audio_bytes]},
    {"task":"video", "data":[dog_video_bytes]},
]

for case in cases:
    prediction = httpx.post(
        f"{REMOTE_URL}/inference",
        headers={"X-API-Key": TOKEN},
        data=msgpack.packb(case),
        timeout=httpx.Timeout(timeout=60.0)
    )
    if prediction.status_code == 200:
        data = msgpack.unpackb(prediction.content)
        print(f'{str(data)[:50]}...{str(data)[-50:]}')
    else:
        print(prediction.status_code)
        print(prediction.content)