import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoImageProcessor, AutoModelForImageClassification
from src.config import CACHE_DIR, MODEL_NAME
from rich.console import Console
from src.vexpert import VisionExpert

from src.config import (
    EXPERT,
    LORA,
    L_TRAINABLE,
    V_TRAINABLE,
)

def load_model_and_processor():
    
    console = Console()

    if LORA:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
        console.print("[yellow2]    LoRA mode on.! 🛰️[/yellow2]")
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
        console.print("[yellow2]    LoRA mode off.! 🛰️[/yellow2]")
    
    model.config.use_cache = False

    if L_TRAINABLE:
        console.print("[red]    The language module is on fire![/red] 🔥")
    else:
        for param in model.language_model.parameters():
            param.requires_grad = False
        console.print("[cyan]    The langauge module is frozen![/cyan] 🧊")
    
    if V_TRAINABLE:
        console.print("[red]    The vision module is on fire![/red] 🔥")
    else:
        for param in model.vision_model.parameters():
            param.requires_grad = False
        console.print("[cyan]    The vision module is frozen![/cyan] 🧊")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR
    )

    if EXPERT:
        print("Initialisng vision expert.")
        vision_expert = VisionExpert(load_dir=CACHE_DIR)
        console.print("[yellow2]    The vision expert has been loaded![/yellow2] 👁️")
    else:
        vision_expert = False
        console.print("[yellow2]    No vision expert has been loaded![/yellow2] 👁️")
        
    return model, processor, vision_expert
