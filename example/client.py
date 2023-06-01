from typing import List, Literal, TypedDict
import modelz

with open("dog_audio.wav", "rb") as f:
    dog_audio_bytes = f.read()

with open("dog_image.jpg", "rb") as f:
    dog_image_bytes = f.read()

with open("dog_video.mp4", "rb") as f:
    dog_video_bytes = f.read()

# Change to real deployment key and token
DEPLOYMENT_KEY = "imagebind-zzzzzzzzzzzzzzzz"
TOKEN = "mzi-ttt"


class Input(TypedDict):
    input: List[bytes | str]
    model: Literal[
        "imagebind-image",
        "imagebind-audio",
        "imagebind-video",
        "imagebind-text",
    ]


class Embedding(TypedDict):
    embedding: List[float]
    index: int
    object: Literal["embedding"]


class Output(TypedDict):
    data: List[Embedding]
    model: Literal[
        "imagebind-image",
        "imagebind-audio",
        "imagebind-video",
        "imagebind-text",
    ]
    object: Literal["list"]


cases: List[Input] = [
    {"model": "imagebind-text", "input": ["A dog", "doggery", "puppy"]},
    {"model": "imagebind-image", "input": [dog_image_bytes]},
    {"model": "imagebind-audio", "input": [dog_audio_bytes]},
    {"model": "imagebind-video", "input": [dog_video_bytes]},
]

client = modelz.ModelzClient(key=TOKEN, deployment=DEPLOYMENT_KEY, timeout=60)

for case in cases:
    print("case for task {}:".format(case["model"]))
    prediction: Output = client.inference(case, serde="msgpack").data
    embeddings = prediction["data"]
    for emb in embeddings:
        print(
            f"{emb['index']}-{str(emb['embedding'])[:50]}...{str(emb['embedding'])[-50:]}"
        )
