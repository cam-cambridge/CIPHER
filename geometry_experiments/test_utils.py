from transformers import AutoModelForVision2Seq, AutoProcessor
import re
import math

CACHE_DIR = "/home/cm2161/rds/hpc-work/"

def load_model_and_processor(experiment):

    if experiment["lora"]:
        model = AutoModelForVision2Seq.from_pretrained(
            experiment['path'],
            device_map="auto",
            cache_dir=CACHE_DIR
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            experiment['path'],
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
    
    processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        cache_dir=CACHE_DIR,
    )
    
    if experiment.get('expert'):
        from src.vexpert import VisionExpert
        print("Loading vision expert")
        vision_expert = VisionExpert(load_dir=CACHE_DIR)
    else:
        vision_expert = False

    return model, processor, vision_expert


def val_collate_fn_text(example, processor):
    
    template= format_data(example)

    texts = processor.apply_chat_template(template["messages"], tokenize=False)

    batch = processor(text=texts, return_tensors="pt", padding=True)

    return batch


def format_data(example):
    
    formatted_data = {"messages": [{"role": "user","content": []}]}

    formatted_data["messages"][0]["content"].append(
        {
            "type": "text", "text": example
        }
    )

    return formatted_data


def answer_extractor(examples, start_marker="<|start_header_id|>assistant<|end_header_id|>", end_marker="<|eot_id|>"):
    """
    Extracts content between start_marker and end_marker in the input strings.
    
    Args:
        examples (str or list of str): A single string or list of strings to process.
        start_marker (str): The starting marker to look for
        end_marker (str): The ending marker to look for
        
    Returns:
        list of str: Extracted content from each input string.
    """

    if isinstance(examples, str):
        examples = [examples]  # Convert single string to list for uniform processing
    
    extracted_content = []
    for example in examples:
        # Escape special regex characters in markers
        escaped_start = re.escape(start_marker)
        escaped_end = re.escape(end_marker)
        pattern = f"{escaped_start}\\n\\n(.*?)({escaped_end}.*)?$"
        match = re.search(pattern, example, re.DOTALL)
        extracted_content.append(match.group(1) if match else None)

    return extracted_content

def calculate_extrusion(x1, y1, x2, y2, previous_e):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    extrusion = distance * 0.04
    return previous_e + extrusion

def start_gcode():
    return """
        G21 ; Set units to millimeters
        G90 ; Use absolute positioning
        M104 S200 ; Set extruder temperature to 200C
        M140 S60 ; Set bed temperature to 60C
        G28 ; Home all axes
        M109 S200 ; Wait for extruder to reach 200C
        M190 S60 ; Wait for bed to reach 60C
        G92 E0 ; Reset extruder
        G1 Z0.2 F300 ; Move to first layer height
        G1 X110 Y100 F1500 ; Move to start position
        G1 F9000
    """

def end_gcode():
    return """
        G1 Z50 F300 ; Lift nozzle
        M104 S0 ; Turn off extruder
        M140 S0 ; Turn off bed
        G28 X0 Y0 ; Home axes
        M84 ; Disable motors
    """

def save_gcode(filename, gcode):
    with open(filename, 'w') as f:
        f.write(gcode)
