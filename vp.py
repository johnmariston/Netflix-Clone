import mediapipe as mp
import cv2
import numpy as np
import time
import os

# Create a folder for screenshots
screenshot_folder = "Screenshots"
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

# Constants
ml = 150
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 3
prevx, prevy = 0, 0
smooth_factor = 0.3
selected_color = (0, 0, 0)  # Default: Black

# Undo/Redo storage
history = []
redo_stack = []
history_limit = 10

def save_state():
    """Saves the current mask state for undo."""
    if len(history) >= history_limit:
        history.pop(0)
    history.append(mask.copy())
    redo_stack.clear()  # Clear redo stack when a new action is made

def undo():
    """Restores the last saved state if available."""
    if history:
        redo_stack.append(mask.copy())  # Save current state for redo
        mask[:] = history.pop()

def redo():
    """Restores the last undone state if available."""
    if redo_stack:
        history.append(mask.copy())  # Save current state for undo
        mask[:] = redo_stack.pop()

def take_screenshot():
    """Saves the current screen as an image file."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(screenshot_folder, f"screenshot_{timestamp}.png")
    cv2.imwrite(filename, mask)
    print(f"✅ Screenshot saved: {filename}")

def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

def index_raised(yi, y9):
    return (y9 - yi) > 40

hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Load tool icons
tools = cv2.imread("tools.png").astype('uint8')
mask = np.ones((480, 640, 3), dtype='uint8') * 255  # 3-channel image for colors

cap = cv2.VideoCapture(0)
cv2.namedWindow("paint app", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("paint app", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

            # Apply smoothing
            prevx = int(prevx + smooth_factor * (x - prevx))
            prevy = int(prevy + smooth_factor * (y - prevy))

            if ml < x < max_x and y < max_y:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()
                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1
                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Your current tool set to:", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
            y9 = int(i.landmark[9].y * 480)

            if curr_tool == "draw" and index_raised(yi, y9):
                save_state()  # Save state before drawing
                cv2.line(mask, (prevx, prevy), (x, y), selected_color, thick)
            elif curr_tool in ["line", "rectangle", "circle"]:
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    if curr_tool == "line":
                        cv2.line(frm, (xii, yii), (x, y), selected_color, thick)
                    elif curr_tool == "rectangle":
                        cv2.rectangle(frm, (xii, yii), (x, y), selected_color, thick)
                    elif curr_tool == "circle":
                        cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), selected_color, thick)
                else:
                    if var_inits:
                        save_state()
                        if curr_tool == "line":
                            cv2.line(mask, (xii, yii), (x, y), selected_color, thick)
                        elif curr_tool == "rectangle":
                            cv2.rectangle(mask, (xii, yii), (x, y), selected_color, thick)
                        elif curr_tool == "circle":
                            cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), selected_color, thick)
                        var_inits = False
            elif curr_tool == "erase" and index_raised(yi, y9):
                save_state()
                cv2.circle(mask, (x, y), 30, (255, 255, 255), -1)

    # Merge mask with frame
    blended = cv2.addWeighted(frm, 0.7, mask, 0.3, 0)

    # Draw UI
    blended[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, blended[:max_y, ml:max_x], 0.3, 0)
    cv2.putText(blended, curr_tool, (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(blended, f"Color: {selected_color}", (270 + ml, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, selected_color, 2)

    cv2.imshow("paint app", blended)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break
    elif key == ord('c'):  # Clear screen
        mask[:] = 255
    elif key == ord('z'):  # Undo
        undo()
    elif key == ord('y'):  # Redo
        redo()
    elif key == ord('s'):  # Take Screenshot
        take_screenshot()
    elif key == ord('r'):  # Red color
        selected_color = (0, 0, 255)
    elif key == ord('g'):  # Green color
        selected_color = (0, 255, 0)
    elif key == ord('b'):  # Blue color
        selected_color = (255, 0, 0)
    elif key == ord('k'):  # Black color
        selected_color = (0, 0, 0)

cap.release()
cv2.destroyAllWindows()
