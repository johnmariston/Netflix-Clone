This project is a **virtual paint application** that lets you draw on the screen using your hand gestures detected by a webcam. 

[Tool Preview](tools.png)

 ✨ Features
- ✍️ Draw using hand gestures
- 📏 Tools: Line, Rectangle, Circle, Free Draw, Eraser
- ♻️ Undo and Redo support
- 📸 Screenshot your drawings (`s` key)
- 🎨 Change colors using keyboard shortcuts
- 🔄 Smooth and accurate drawing with MediaPipe hand tracking

🛠 Requirements
- Python 3.x
- `opencv-python`
- `mediapipe`
- `numpy`

Install using pip:
```bash
pip install opencv-python mediapipe numpy

🎮 Controls

r   - Red color  
g   - Green color  
b   - Blue color  
k   - Black color  
z   - Undo  
y   - Redo  
s   - Save screenshot  
c   - Clear screen  
Esc - Exit application  
