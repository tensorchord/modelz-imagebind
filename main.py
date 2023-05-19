from typing import Dict, List, Literal, TypedDict
from io import BytesIO
import torch
from mosec import Server, Worker, get_logger
from mosec.mixin import MsgpackMixin

import data
from models import imagebind_model
from models.imagebind_model import ModalityType

logger = get_logger()

class Input(TypedDict):
    task: Literal["image", "audio", "video", "text"]
    data: List[bytes | str]

TASK_TYPE_DATA = {
    "image": (ModalityType.VISION, lambda paths, device: data.load_and_transform_vision_data(paths, device)),
    "audio": (ModalityType.AUDIO, lambda paths, device: data.load_and_transform_audio_data(paths, device)),
    "video": (ModalityType.VISION, lambda paths, device: data.load_and_transform_video_data(paths, device)),
    "text": (ModalityType.TEXT, lambda text, device: data.load_and_transform_text(text, device)),
}

class ImageBind(MsgpackMixin, Worker):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = imagebind_model.imagebind_huge(pretrained=True)

        self.pipe = self.pipe.to(self.device)
        self.pipe.eval()

    def forward(self, input_data: Input) -> List[List[float]]:
        inputs: Dict[ModalityType, torch.Tensor] = {}

        task = input_data["task"]
        if task not in TASK_TYPE_DATA.keys():
            raise RuntimeError(f"unrecognized task: {task}")
        task_label, data_generator = TASK_TYPE_DATA[task]

        if task != 'text':
            paths = [BytesIO(d) for d in input_data["data"]]
            inputs[task_label] = data_generator(paths, self.device)
        else:
            inputs[task_label] = data_generator(input_data["data"], self.device)

        with torch.no_grad():
            embeddings = self.pipe(inputs)

        return embeddings[task_label].cpu().tolist()


if __name__ == "__main__":
    server = Server()
    server.append_worker(ImageBind, num=1, max_batch_size=1)
    server.run()
