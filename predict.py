import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import utils
from ldm.util import instantiate_from_config
from cog import BasePredictor, Input, Path



MODEL_CONFIG_PATH = "models/gqa_inpaint/ldm/config.yaml"
MODEL_CKPT_PATH = "models/gqa_inpaint/ldm/model.ckpt"

# Url to the weights.tar. It also contains config.yaml and tokenizer config
WEIGHTS_URL = "https://weights.replicate.delivery/default/inst-inpaint/weigts.tar" 
if not os.path.exists(MODEL_CKPT_PATH):
    print("Downloading checkpoints and config...")
    utils.download_model(WEIGHTS_URL, "/src//models")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        parsed_config = OmegaConf.load(MODEL_CONFIG_PATH)
        
        self.model = instantiate_from_config(parsed_config["model"])
        model_state_dict = torch.load(MODEL_CKPT_PATH)["state_dict"]
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        self.model.to(self.device)
        print("Model loaded successfully!")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        instruction: str = Input(
            description="Text instruction describing which object/s to remove",
        ),
        center_crop: bool = Input(
            description="Whether to center crop the image",
            default=True, 
        ),
        num_inference_steps: int = Input(
            description="Number of diffusion steps",  
            default=30, ge=0, le=100,
        ),
        seed: int = Input(
            description="Seed for the diffusion process. If not provided, it will be randomized.", 
            default=None, ge=0, le=100,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        pil_image = Image.open(image).convert("RGB")
        if seed is None:
            seed = np.random.randint(0, 1000)

        tensor_image, _ = utils.preprocess_image(pil_image, center_crop=center_crop)
        
        output = self.model.inpaint(
            image=tensor_image,
            instruction=instruction,
            num_steps=num_inference_steps,
            device=self.device,
            return_pil=True,
            seed=seed
        )
        
        output_path = "/tmp/output.png"
        output.save(output_path)

        return Path(output_path)