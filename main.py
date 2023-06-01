from typing import Any, Dict, List, Literal
from io import BytesIO
import msgspec
import torch
from mosec import Server, Worker, get_logger
from mosec.mixin import MsgpackMixin

import data
from models import imagebind_model
from models.imagebind_model import ModalityType
from llmspec import EmbeddingRequest, EmbeddingResponse

logger = get_logger()


class ImageBindRequest(EmbeddingRequest):
    model: Literal[
        "imagebind-image", "imagebind-audio", "imagebind-video", "imagebind-text"
    ]
    input: List[bytes] | List[str]

    @classmethod
    def from_dict(cls, buf: Dict[str, Any]):
        if buf["model"] == "imagebind-text":
            input = msgspec.from_builtins(buf["input"], type=List[str])
        else:
            input = msgspec.from_builtins(buf["input"], type=List[bytes])
        return cls(model=buf["model"], input=input)


MODEL_TYPE_HANDLER = {
    "imagebind-image": (
        ModalityType.VISION,
        lambda paths, device: data.load_and_transform_vision_data(paths, device),
    ),
    "imagebind-audio": (
        ModalityType.AUDIO,
        lambda paths, device: data.load_and_transform_audio_data(paths, device),
    ),
    "imagebind-video": (
        ModalityType.VISION,
        lambda paths, device: data.load_and_transform_video_data(paths, device),
    ),
    "imagebind-text": (
        ModalityType.TEXT,
        lambda text, device: data.load_and_transform_text(text, device),
    ),
}


class ImageBindResponse(EmbeddingResponse):
    model: Literal[
        "imagebind-image",
        "imagebind-audio",
        "imagebind-video",
        "imagebind-text",
    ]


class ImageBind(MsgpackMixin, Worker):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = imagebind_model.imagebind_huge(pretrained=True)

        self.pipe = self.pipe.to(self.device)
        self.pipe.eval()

    def deserialize(self, buf: bytes) -> ImageBindRequest:
        data: Dict[str, Any] = super().deserialize(buf)
        return ImageBindRequest.from_dict(data)

    def forward(self, req: ImageBindRequest) -> ImageBindResponse:
        inputs: Dict[ModalityType, torch.Tensor] = {}

        task = req.model
        if task not in MODEL_TYPE_HANDLER.keys():
            raise RuntimeError(f"unrecognized task: {task}")
        task_label, data_generator = MODEL_TYPE_HANDLER[task]

        input_data = [req.input] if not isinstance(req.input, list) else req.input
        if task != "imagebind-text":
            paths = [BytesIO(d) for d in input_data]
            inputs[task_label] = data_generator(paths, self.device)
        else:
            inputs[task_label] = data_generator(input_data, self.device)

        with torch.no_grad():
            embeddings = self.pipe(inputs)
        embeddings_detach = embeddings[task_label].cpu().tolist()
        output: ImageBindResponse = {
            "data": [
                {"embedding": emb, "index": i, "object": "embedding"}
                for i, emb in enumerate(embeddings_detach)
            ],
            "model": task,
            "object": "list",
        }
        return output


if __name__ == "__main__":
    server = Server()
    server.append_worker(ImageBind, num=1, max_batch_size=1)
    server.run()
