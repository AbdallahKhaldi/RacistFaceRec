# Facial Recognition System - Group Detection

This system detects faces and identifies which group they belong to: **Group A** or **Group J**.

## Quick Start

### Starting the Server

```bash
chmod +x start.sh stop.sh
./start.sh
```

Or manually:
```bash
source venv/bin/activate
python backend/app.py
```

The server will start at: http://127.0.0.1:5001

### Stopping the Server

```bash
./stop.sh
```

Or manually:
```bash
# Find the process
lsof -ti:5001

# Kill it (replace PID with the number from above)
kill -9 PID
```

Or simply press `CTRL+C` in the terminal where the server is running.

## Features

### 1. Register Faces
- Select Group A or Group J
- Enter a name (for internal tracking only)
- Click "Register Face" to capture from webcam

### 2. Batch Upload
- Select group (A or J)
- Click "Batch Upload"
- Select multiple images (filename becomes the name)
- All images registered to selected group

### 3. Real-Time Detection
- Click "Real-Time Detection"
- Camera continuously scans for faces
- Displays: **"GROUP A"** or **"GROUP J"** (names hidden)
- Click "Stop Real-Time" to end

### 4. Manual Recognition
- Click "Recognize Face" to check a single face
- Shows which group the person belongs to

## Group Colors

- **Group A**: Blue color
- **Group J**: Green color

## Troubleshooting

### Port Already in Use
If you get "Port 5001 is in use" error:
```bash
./stop.sh
./start.sh
```

### Camera Not Working

**Important**: Always access the app via `http://127.0.0.1:5001` or `http://localhost:5001` (NOT the IP address like 10.x.x.x)

Common solutions:

1. **Check browser permissions**
   - Look for a camera icon in the address bar
   - Click it and allow camera access
   - Refresh the page

2. **Camera already in use**
   - Close other apps using the camera (Zoom, Skype, etc.)
   - Close other browser tabs with camera access

3. **Browser compatibility**
   - Use Chrome, Firefox, Edge, or Safari
   - Make sure browser is up to date

4. **Hard refresh the page**
   - Mac: `Cmd + Shift + R`
   - Windows/Linux: `Ctrl + Shift + R`

5. **Check browser console**
   - Press F12 to open developer tools
   - Look at Console tab for error messages

6. **macOS Camera Permissions**
   - Go to System Preferences > Security & Privacy > Camera
   - Make sure your browser has camera access enabled

### Dependencies Missing
```bash
source venv/bin/activate
pip install -r backend/requirements.txt
```
