from tkinter import *
import cv2
from PIL import Image, ImageTk

# Variables to store the coordinates of the four points (corners)
polygons = []
points = [(100, 100), (300, 100), (300, 300), (100, 300)]
dragging_point = None

def draw_figure(event):
    global points, dragging_point

    if dragging_point is not None:
        # Update the coordinates of the selected point
        points[dragging_point] = (event.x, event.y)

        # Redraw the figure with updated points
        canvas.coords(figure, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], points[3][0], points[3][1])
        for i, point in enumerate(points):
            canvas.coords(handles[i], point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5)

def start_draw(event):
    global points, dragging_point
    # Check if the user clicked near any point and start dragging that point
    for i, point in enumerate(points):
        if is_near(event.x, event.y, point[0], point[1]):
            dragging_point = i
            break


def is_near(x1, y1, x2, y2, threshold=10):
    """Check if (x1, y1) is near (x2, y2) within a threshold."""
    return abs(x1 - x2) < threshold and abs(y1 - y2) < threshold


def save_coordinates():
    """Save the coordinates of all four points."""
    print("Figure coordinates:")
    with open("polygons.txt", "a") as f:
        f.write(f"{points}\n")
    for i, point in enumerate(points):
        print(f"Point {i+1}: {point}")


def display_first_frame(video_path):
    # Create the Tkinter root window first
    root = Tk()
    root.geometry("1200x600")
    root.title("Draw and Manipulate 4-Point Figure on Image")

    # Initialize the canvas
    global canvas, figure, handles
    canvas = Canvas(root, width=1020, height=500)
    canvas.pack()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Convert the frame to a format tkinter can handle (PIL image)
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_pil = image_pil.resize((1020, 500), Image.Resampling.LANCZOS)

    # Convert the PIL image to a Tkinter PhotoImage
    image_tk = ImageTk.PhotoImage(image_pil)

    # Add the image to the canvas
    canvas.create_image(0, 0, anchor=NW, image=image_tk)

    # Create the figure (initially a quadrilateral)
    global points
    figure = canvas.create_polygon(points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], points[3][0], points[3][1], outline="red", fill='', width=2)

    # Create draggable corner circles
    handles = [canvas.create_oval(point[0]-5, point[1]-5, point[0]+5, point[1]+5, fill="blue") for point in points]

    # Bind mouse events for figure manipulation
    canvas.bind("<ButtonPress-1>", start_draw)  # When mouse is clicked
    canvas.bind("<B1-Motion>", draw_figure)  # When mouse is dragged

    # Button to save figure coordinates
    save_button = Button(root, text="Save Coordinates", command=save_coordinates)
    save_button.pack()

    # Run the Tkinter main loop
    root.mainloop()

# Replace with your video path
video_path = "/Users/alex/Desktop/algorithms/violette-main/samples/sample.mp4"
display_first_frame(video_path)
