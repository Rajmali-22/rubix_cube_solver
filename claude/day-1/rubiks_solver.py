"""
Rubik's Cube Solver with Computer Vision and 6-Motor Control
Hardware: 6 NEMA-17 steppers (Fixed Clockwise Only)
Communication: Serial (COM3, 9600 baud)
"""

import cv2
import numpy as np
import serial
import time
import kociemba
from collections import Counter

# ============================================================================
# PART A: COMPUTER VISION - COLOR DETECTION
# ============================================================================

class ColorDetector:
    """
    Detects and identifies the 9 stickers on each Rubik's Cube face.
    Uses HSV color space for robust detection under varying lighting.
    """
    
    def __init__(self):
        # HSV ranges for each color (Hue, Saturation, Value)
        # TUNING GUIDE: Adjust these based on your lighting conditions
        # - Increase lower bounds if detecting too many false positives
        # - Decrease upper bounds if missing valid stickers
        # - Use HSV color picker tools to find exact ranges
        
        self.color_ranges = {
            'W': ([0, 0, 200], [180, 30, 255]),      # White: Low saturation, high value
            'Y': ([20, 100, 100], [30, 255, 255]),   # Yellow: Hue ~25
            'R': ([0, 100, 100], [10, 255, 255]),    # Red: Hue 0-10 (wraps around)
            'O': ([10, 100, 100], [20, 255, 255]),   # Orange: Hue 10-20
            'B': ([100, 100, 100], [130, 255, 255]), # Blue: Hue 100-130
            'G': ([40, 100, 100], [80, 255, 255])    # Green: Hue 40-80
        }
        
        # Alternative Red range (for wraparound at 180)
        self.red_alt = ([170, 100, 100], [180, 255, 255])
        
        self.cap = None
        self.face_map = {}  # Store detected faces
        
    def init_camera(self, camera_id=0):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def detect_stickers(self, frame):
        """
        Detect 9 sticker positions on current frame.
        Returns list of (x, y, color) for each sticker.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define 3x3 grid regions (adjust based on your setup)
        h, w = frame.shape[:2]
        margin = 50
        grid_w = (w - 2*margin) // 3
        grid_h = (h - 2*margin) // 3
        
        stickers = []
        
        for row in range(3):
            for col in range(3):
                # Calculate region of interest (ROI)
                x1 = margin + col * grid_w
                y1 = margin + row * grid_h
                x2 = x1 + grid_w
                y2 = y1 + grid_h
                
                roi = hsv[y1:y2, x1:x2]
                
                # Detect dominant color in this ROI
                color = self.identify_color(roi)
                
                # Draw rectangle and label
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, color, (center_x-10, center_y+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                stickers.append((row, col, color))
        
        return stickers
    
    def identify_color(self, roi):
        """
        Identify the dominant color in a region of interest.
        Returns single character: W, Y, R, O, B, G
        """
        best_match = 'W'  # Default to white
        max_pixels = 0
        
        for color_code, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(roi, lower, upper)
            pixel_count = cv2.countNonZero(mask)
            
            # Special handling for red (wraparound)
            if color_code == 'R':
                lower_alt = np.array(self.red_alt[0])
                upper_alt = np.array(self.red_alt[1])
                mask_alt = cv2.inRange(roi, lower_alt, upper_alt)
                pixel_count += cv2.countNonZero(mask_alt)
            
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                best_match = color_code
        
        return best_match
    
    def calibrate_face(self, face_name):
        """
        Interactive calibration for one face.
        face_name: U, D, L, R, F, B
        """
        print(f"\n=== Calibrating {face_name} Face ===")
        print("Position the cube so the camera sees the face clearly.")
        print("Press SPACE to capture, ESC to cancel")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect and draw stickers
            stickers = self.detect_stickers(frame)
            
            cv2.imshow(f'Calibrate {face_name} Face', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return None
            elif key == 32:  # SPACE
                # Extract color sequence (row-major order)
                colors = [s[2] for s in sorted(stickers, key=lambda x: (x[0], x[1]))]
                cv2.destroyAllWindows()
                return colors
    
    def calibrate_all_faces(self):
        """
        Guide user through calibrating all 6 faces.
        Returns 54-character string in Kociemba format.
        """
        self.init_camera()
        
        # Order matters! Kociemba expects: U, R, F, D, L, B
        face_order = ['U', 'R', 'F', 'D', 'L', 'B']
        face_names = {
            'U': 'Up (White/Yellow)',
            'R': 'Right',
            'F': 'Front',
            'D': 'Down (White/Yellow)',
            'L': 'Left',
            'B': 'Back'
        }
        
        all_colors = []
        center_map = {}  # Map detected color -> face letter
        
        for face in face_order:
            colors = self.calibrate_face(f"{face} - {face_names[face]}")
            if colors is None:
                self.cap.release()
                cv2.destroyAllWindows()
                raise Exception("Calibration cancelled")
            
            # Store center sticker to build color mapping
            center_color = colors[4]  # Center is index 4 in 3x3 grid
            center_map[center_color] = face
            print(f"Detected center: {center_color} -> Face {face}")
            
            all_colors.extend(colors)
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Convert detected colors (W,Y,R,O,B,G) to Kociemba faces (U,R,F,D,L,B)
        raw_string = ''.join(all_colors)
        print(f"\nRaw detected: {raw_string}")
        print(f"Color mapping: {center_map}")
        
        # Translate using center map
        kociemba_string = ''.join([center_map.get(c, c) for c in raw_string])
        print(f"Kociemba format: {kociemba_string}")
        
        return kociemba_string


# ============================================================================
# PART B: CUBE SOLVER - KOCIEMBA LOGIC
# ============================================================================

class CubeSolver:
    """
    Validates and solves the Rubik's Cube using Kociemba algorithm.
    """
    
    @staticmethod
    def validate_cube_string(cube_string):
        """
        Validate that the 54-character string is valid:
        - Exactly 54 characters
        - Exactly 9 of each color (W, Y, R, O, B, G)
        """
        if len(cube_string) != 54:
            raise ValueError(f"Invalid length: {len(cube_string)} (expected 54)")
        
        color_counts = Counter(cube_string)
        required_colors = {'W', 'Y', 'R', 'O', 'B', 'G'}
        
        # Check each color appears exactly 9 times
        for color in required_colors:
            count = color_counts.get(color, 0)
            if count != 9:
                raise ValueError(f"Color {color} appears {count} times (expected 9)")
        
        # Check no invalid colors
        invalid = set(cube_string) - required_colors
        if invalid:
            raise ValueError(f"Invalid colors detected: {invalid}")
        
        print("✓ Cube validation passed")
        return True
    
    @staticmethod
    def solve(cube_string, max_depth=24, timeout=10):
        """
        Solve the cube using Kociemba algorithm.
        
        Args:
            cube_string: 54-char string with face notation (U,R,F,D,L,B)
            max_depth: Maximum search depth (default 24, increase for harder cubes)
            timeout: Timeout in seconds (default 10)
        
        Returns solution string (e.g., "D2 R' L B U2 F")
        """
        try:
            # Kociemba signature: solve(cubestring, patternstring="", max_depth=24)
            # patternstring is optional (for pattern solving, we don't need it)
            solution = kociemba.solve(cube_string, max_depth=max_depth)
            print(f"\n✓ Solution found: {solution}")
            return solution
        except ValueError as e:
            # More detailed error for invalid cube states
            raise Exception(
                f"Invalid cube configuration: {e}\n"
                f"This usually means:\n"
                f"  - Color detection errors (recheck your calibration)\n"
                f"  - Physically impossible state (cube was disassembled incorrectly)\n"
                f"  - Wrong center color mapping"
            )
        except Exception as e:
            raise Exception(f"Kociemba solver failed: {e}")


# ============================================================================
# PART C: MOTOR CONTROLLER - SERIAL COMMUNICATION
# ============================================================================

class MotorController:
    """
    Handles serial communication with Arduino.
    Translates Kociemba moves to Fixed-Clockwise commands.
    """
    
    def __init__(self, port='COM3', baudrate=9600, timeout=5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        
    def connect(self):
        """Establish serial connection and wait for Arduino ready"""
        print(f"Connecting to {self.port}...")
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2)  # Wait for Arduino to reset
        
        # Wait for "READY" signal from Arduino
        ready = self.ser.readline().decode().strip()
        if "READY" in ready:
            print("✓ Arduino connected and ready")
        else:
            print(f"Warning: Expected READY, got: {ready}")
    
    def translate_move(self, move):
        """
        Translate Kociemba notation to Fixed-Clockwise format.
        
        Examples:
        R   -> R1  (90° CW)
        R2  -> R2  (180° CW)
        R'  -> R3  (270° CW = 90° CCW)
        U   -> U1
        D2  -> D2
        """
        move = move.strip()
        
        if not move:
            return None
        
        face = move[0]  # U, D, L, R, F, B
        
        if len(move) == 1:
            # Standard move (90° CW)
            return f"{face}1"
        elif move[1] == '2':
            # Double move (180° CW)
            return f"{face}2"
        elif move[1] == "'":
            # Prime/inverse move (270° CW = 90° CCW)
            return f"{face}3"
        else:
            raise ValueError(f"Invalid move notation: {move}")
    
    def execute_solution(self, solution_string):
        """
        Parse solution and send moves to Arduino one at a time.
        Implements handshake protocol: Send -> Wait for OK -> Next
        """
        moves = solution_string.split()
        total_moves = len(moves)
        
        print(f"\n=== Executing {total_moves} moves ===")
        
        for i, move in enumerate(moves, 1):
            translated = self.translate_move(move)
            
            if translated is None:
                continue
            
            print(f"[{i}/{total_moves}] Sending: {move} -> {translated}")
            
            # Send command
            self.ser.write(f"{translated}\n".encode())
            
            # Wait for "OK" acknowledgment
            response = self.ser.readline().decode().strip()
            
            if response == "OK":
                print(f"  ✓ Confirmed")
            else:
                print(f"  ✗ Unexpected response: {response}")
                raise Exception(f"Arduino did not acknowledge move {translated}")
            
            time.sleep(0.1)  # Small delay between moves
        
        print("\n✓ All moves executed successfully")
    
    def disconnect(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main workflow:
    1. Detect cube state with CV
    2. Validate and solve with Kociemba
    3. Execute solution via Serial
    """
    
    print("="*60)
    print("RUBIK'S CUBE SOLVER - 6-MOTOR SYSTEM")
    print("="*60)
    
    try:
        # Step 1: Computer Vision
        print("\n[STEP 1] Computer Vision - Calibrating Cube State")
        detector = ColorDetector()
        cube_string = detector.calibrate_all_faces()
        
        # Step 2: Solve
        print("\n[STEP 2] Solving Cube")
        CubeSolver.validate_cube_string(cube_string)
        solution = CubeSolver.solve(cube_string)
        
        # Step 3: Execute
        print("\n[STEP 3] Executing Solution")
        controller = MotorController(port='COM3', baudrate=9600)
        controller.connect()
        controller.execute_solution(solution)
        controller.disconnect()
        
        print("\n" + "="*60)
        print("✓ CUBE SOLVED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()