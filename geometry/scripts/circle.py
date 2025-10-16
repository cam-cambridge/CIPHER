import math

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

def generate_circle_layer(center_x, center_y, radius, segments=100, start_e=0):
    """Generates G-code for a single perimeter circle."""
    gcode = ""
    angle_step = 2 * math.pi / segments
    e = start_e

    # First point on circle
    x_start = center_x + radius
    y_start = center_y
    gcode += f"G1 X{x_start:.3f} Y{y_start:.3f} F1500\n"

    # Draw the circle perimeter
    for i in range(1, segments + 1):
        angle = i * angle_step
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        e = calculate_extrusion(x_start, y_start, x, y, e)
        gcode += f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
        x_start, y_start = x, y

    return gcode, e

def generate_circle_infill(center_x, center_y, radius, spacing=0.4, start_e=0):
    """Generates linear infill inside the circle."""
    gcode = ""
    e = start_e
    y = center_y - radius
    while y <= center_y + radius:
        dx = math.sqrt(radius**2 - (y - center_y)**2)
        x1 = center_x - dx
        x2 = center_x + dx
        gcode += f"G1 X{x1:.3f} Y{y:.3f} F1500\n"
        e = calculate_extrusion(x1, y, x2, y, e)
        gcode += f"G1 X{x2:.3f} Y{y:.3f} E{e:.5f}\n"
        y += spacing
    return gcode, e

def generate_circle(center_x, center_y, radius, layers, start_e=0):
    gcode = start_gcode()
    for layer in range(layers):
        gcode += "G1 Z" + str(layer * 0.2) + " F300\n"
        gcode, e = generate_circle_layer(center_x, center_y, radius, start_e=start_e)
        gcode += "G92 E0\n"
        gcode += generate_circle_infill(center_x, center_y, radius, start_e=e)[0]
        gcode += "G92 E0\n"
    gcode += end_gcode()
    return gcode

def main():
    filename = "circle.gcode"
    center_x = 110
    center_y = 100
    radius = 10
    layers = 10
    gcode = generate_circle(center_x, center_y, radius, layers)
    save_gcode(filename, gcode)
    print(f"Generated G-code saved to {filename}")

if __name__ == "__main__":
    main()