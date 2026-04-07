from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import kociemba
from typing import List, Optional
import io
from PIL import Image

app = FastAPI(title="Rubix Cube Solver API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_base64: str  # Data URL or raw base64

class SolveRequest(BaseModel):
    cube_string: str  # 54-char string (U, R, F, D, L, B)

# --- ADVANCED COLOR DETECTION ---

class ColorAnalyzer:
    def __init__(self):
        # Professional HSV ranges (can be auto-calibrated later)
        self.color_ranges = {
            'W': ([0, 0, 150], [180, 50, 255]),      # White
            'Y': ([20, 70, 70], [35, 255, 255]),     # Yellow
            'R1': ([0, 100, 70], [10, 255, 255]),    # Red (range 1)
            'R2': ([160, 100, 70], [180, 255, 255]), # Red (range 2)
            'O': ([10, 100, 70], [20, 255, 255]),    # Orange
            'B': ([90, 80, 70], [130, 255, 255]),    # Blue
            'G': ([40, 70, 70], [90, 255, 255])      # Green
        }

    def decode_image(self, base64_str: str):
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def detect_stickers(self, frame):
        """
        Advanced detection using contour analysis to find the 3x3 grid.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Pre-processing for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        
        # 2. Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, sw, sh = cv2.boundingRect(approx)
                aspect_ratio = float(sw) / sh
                area = cv2.contourArea(cnt)
                
                # Filter by area (relative to frame size) and aspect ratio (square-ish)
                if 1000 < area < (h * w / 10) and 0.8 < aspect_ratio < 1.2:
                    candidates.append((x, y, sw, sh, cnt))

        # 3. Heuristic: If we can't find 9 squares clearly, fallback to a fixed grid center
        if len(candidates) < 9:
            # Fixed grid logic (centered)
            margin = 80
            grid_size = min(h, w) - 2 * margin
            s_size = grid_size // 3
            stickers_roi = []
            for row in range(3):
                for col in range(3):
                    x = (w - grid_size) // 2 + col * s_size
                    y = (h - grid_size) // 2 + row * s_size
                    stickers_roi.append((x, y, s_size, s_size))
        else:
            # Sort candidates to find the most likely 3x3 cluster
            # This is a simplified version; in high-end, we'd use a grid fitting algorithm
            candidates.sort(key=lambda c: (c[1], c[0])) # Sort by y, then x
            # Take the 9 most likely candidates based on area consistency
            candidates = sorted(candidates, key=lambda c: cv2.contourArea(c[4]), reverse=True)[:9]
            candidates.sort(key=lambda c: (c[1]//(h//6), c[0]//(w//6))) # Grid sort
            stickers_roi = [(c[0], c[1], c[2], c[3]) for c in candidates]

        # 4. Color Identification
        detected_colors = []
        for (x, y, sw, sh) in stickers_roi:
            # Sample the center of the sticker
            padding = sw // 4
            roi = hsv[y+padding:y+sh-padding, x+padding:x+sw-padding]
            if roi.size == 0:
                detected_colors.append('W')
                continue
                
            color = self.match_color(roi)
            detected_colors.append(color)
            
        return detected_colors

    def match_color(self, roi):
        best_match = 'W'
        max_score = 0
        
        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(roi, np.array(lower), np.array(upper))
            score = cv2.countNonZero(mask)
            
            # Combine Red ranges
            if color == 'R1':
                mask2 = cv2.inRange(roi, np.array(self.color_ranges['R2'][0]), np.array(self.color_ranges['R2'][1]))
                score += cv2.countNonZero(mask2)
                color = 'R'
            elif color == 'R2':
                continue # Skip standalone R2
                
            if score > max_score:
                max_score = score
                best_match = color
        
        return best_match

analyzer = ColorAnalyzer()

@app.post("/detect")
async def detect_face(request: ImageRequest):
    try:
        frame = analyzer.decode_image(request.image_base64)
        colors = analyzer.detect_stickers(frame)
        return {"colors": colors}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/solve")
async def solve_cube(request: SolveRequest):
    try:
        # The frontend sends a 54-char string of colors (e.g., 'W', 'R', 'G', 'Y', 'O', 'B')
        # Kociemba standard notation requires face identifiers: U, R, F, D, L, B
        # We map colors to faces based on the center sticker of each face.
        
        cube_str = request.cube_string
        if len(cube_str) != 54:
            raise ValueError(f"Cube string must be 54 characters, got {len(cube_str)}")
            
        # Standard faces order: U, R, F, D, L, B (9 stickers each)
        # Center stickers are at indices: 4, 13, 22, 31, 40, 49
        try:
            color_map = {
                cube_str[4]: 'U',
                cube_str[13]: 'R',
                cube_str[22]: 'F',
                cube_str[31]: 'D',
                cube_str[40]: 'L',
                cube_str[49]: 'B'
            }
        except IndexError:
             raise ValueError("Incomplete cube string provided.")

        # Map each color to its corresponding face letter
        mapped_str = "".join([color_map.get(c, 'X') for c in cube_str])
        
        if 'X' in mapped_str:
            raise ValueError("Inconsistent cube colors: some colors do not match any face center.")
            
        # Solve using Kociemba
        solution = kociemba.solve(mapped_str)
        
        # Expand for ESP32 (as per doc: R2 -> R, R; R' -> R_CCW)
        expanded_moves = []
        for move in solution.split():
            face = move[0]
            if len(move) == 1:
                expanded_moves.append(face)
            elif move[1] == '2':
                expanded_moves.append(face)
                expanded_moves.append(face)
            elif move[1] == "'":
                expanded_moves.append(f"{face}_CCW")
                
        return {
            "solution": solution,
            "moves": expanded_moves,
            "count": len(expanded_moves)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsolvable state: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
