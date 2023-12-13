from diffusers import DDPMPipeline
from PIL import Image
import torch

if __name__ == "__main__":
    model_dir = "./ddpm-pizza-128"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_images_to_generate = 1

    pipeline = DDPMPipeline.from_pretrained(model_dir, use_safetensors=True).to(device)
    images = pipeline(
        batch_size=num_images_to_generate, generator=torch.manual_seed(0)
    ).images

    for idx, img in enumerate(images, start=1):
        img.save(f"pizza-generated-{idx}.jpg")
