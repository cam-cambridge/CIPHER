import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoImageProcessor, AutoModelForImageClassification
from rich.console import Console
from vexpert import VisionExpert

def load_model_and_processor(args):
    
    console = Console()

    if args.lora:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path,
            device_map="auto",
            cache_dir=args.cache_dir,
        )
        console.print("[yellow2]    LoRA mode on.! 🛰️[/yellow2]")
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path,
            device_map="auto",
            cache_dir=args.cache_dir,
        )
        console.print("[yellow2]    LoRA mode off.! 🛰️[/yellow2]")
    
    model.config.use_cache = False

    if args.language:
        console.print("[red]    The language module is on fire![/red] 🔥")
    else:
        for param in model.language_model.parameters():
            param.requires_grad = False
        console.print("[cyan]    The langauge module is frozen![/cyan] 🧊")
    
    if args.vision:
        console.print("[red]    The vision module is on fire![/red] 🔥")
    else:
        for param in model.vision_model.parameters():
            param.requires_grad = False
        console.print("[cyan]    The vision module is frozen![/cyan] 🧊")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir
    )

    if args.expert:
        print("Initialisng vision expert.")
        vision_expert = VisionExpert(load_dir=args.cache_dir)
        console.print("[yellow2]    The vision expert has been loaded![/yellow2] 👁️")
    else:
        vision_expert = False
        console.print("[yellow2]    No vision expert has been loaded![/yellow2] 👁️")
        
    return model, processor, vision_expert
