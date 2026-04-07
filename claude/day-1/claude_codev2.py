"""
Rubik's Cube Solver with Computer Vision and Kociemba Algorithm
Phase 1: Detect cube state and generate solution (no hardware execution yet)
"""

import cv2
import numpy as np
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
        
        self.color_ranges = {
            'W': ([0, 0, 180], [180, 40, 255]),      # White: Low saturation, high value
            'Y': ([20, 80, 80], [35, 255, 255]),     # Yellow: Hue ~25
            'R': ([0, 100, 80], [10, 255, 255]),     # Red: Hue 0-10 (wraps around)
            'O': ([10, 100, 80], [20, 255, 255]),    # Orange: Hue 10-20
            'B': ([90, 80, 80], [130, 255, 255]),    # Blue: Hue 90-130
            'G': ([35, 80, 80], [85, 255, 255])      # Green: Hue 35-85
        }
        
        # Alternative Red range (for wraparound at 180)
        self.red_alt = ([165, 100, 80], [180, 255, 255])
        
        self.cap = None
        self.calibration_mode = "manual"  # manual or auto
        
    def init_camera(self, camera_id=0):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✓ Camera initialized")
        
    def detect_stickers(self, frame):
        """
        Detect 9 sticker positions on current frame.
        Returns list of (row, col, color) for each sticker.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # Define 3x3 grid regions (centered on frame)
        h, w = frame.shape[:2]
        margin = 80
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
                
                # Take smaller center region for more accurate color detection
                padding = 15
                roi = hsv[y1+padding:y2-padding, x1+padding:x2-padding]
                
                # Detect dominant color in this ROI
                color = self.identify_color(roi)
                
                # Draw rectangle and label on original frame
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Color-coded rectangles
                box_colors = {
                    'W': (255, 255, 255), 'Y': (0, 255, 255),
                    'R': (0, 0, 255), 'O': (0, 165, 255),
                    'B': (255, 0, 0), 'G': (0, 255, 0)
                }
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_colors.get(color, (0, 255, 0)), 2)
                cv2.putText(frame, color, (center_x-10, center_y+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                stickers.append((row, col, color))
        
        return stickers
    
    def identify_color(self, roi):
        """
        Identify the dominant color in a region of interest.
        Returns single character: W, Y, R, O, B, G
        """
        if roi.size == 0:
            return 'W'  # Default if ROI is empty
        
        best_match = 'W'
        max_pixels = 0
        
        for color_code, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(roi, lower, upper)
            pixel_count = cv2.countNonZero(mask)
            
            # Special handling for red (wraparound at 180°)
            if color_code == 'R':
                lower_alt = np.array(self.red_alt[0], dtype=np.uint8)
                upper_alt = np.array(self.red_alt[1], dtype=np.uint8)
                mask_alt = cv2.inRange(roi, lower_alt, upper_alt)
                pixel_count += cv2.countNonZero(mask_alt)
            
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                best_match = color_code
        
        return best_match
    
    def calibrate_face(self, face_name, face_description):
        """
        Interactive calibration for one face.
        face_name: U, D, L, R, F, B
        face_description: Human-readable description
        """
        print(f"\n{'='*60}")
        print(f"CALIBRATING: {face_name} Face - {face_description}")
        print(f"{'='*60}")
        print("Instructions:")
        print("  - Position cube so this face is clearly visible")
        print("  - Ensure good lighting (avoid shadows)")
        print("  - Press SPACE when stickers are correctly detected")
        print("  - Press 'R' to retry if colors are wrong")
        print("  - Press ESC to cancel\n")
        
        retry_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("✗ Failed to grab frame")
                break
            
            # Detect and draw stickers
            stickers = self.detect_stickers(frame)
            
            # Extract color sequence (row-major order)
            colors = [s[2] for s in sorted(stickers, key=lambda x: (x[0], x[1]))]
            center_color = colors[4]
            
            # Display info on frame
            info_text = f"Face: {face_name} | Center: {center_color} | Press SPACE to confirm"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Rubiks Cube Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("✗ Calibration cancelled by user")
                return None
            elif key == ord('r') or key == ord('R'):  # Retry
                retry_count += 1
                print(f"  Retrying... (attempt #{retry_count + 1})")
                continue
            elif key == 32:  # SPACE
                print(f"✓ Face {face_name} captured | Center: {center_color}")
                print(f"  Detected: {' '.join(colors)}")
                return colors
    
    def calibrate_all_faces(self):
        """
        Guide user through calibrating all 6 faces.
        Returns 54-character string in Kociemba format (U,R,F,D,L,B letters).
        """
        self.init_camera()
        
        # Order matters! Kociemba expects: U, R, F, D, L, B
        face_order = [
            ('U', 'Up (typically White or Yellow)'),
            ('R', 'Right'),
            ('F', 'Front'),
            ('D', 'Down (typically Yellow or White)'),
            ('L', 'Left'),
            ('B', 'Back')
        ]
        
        all_colors = []
        center_map = {}  # Map detected color (W,Y,R,O,B,G) -> face letter (U,R,F,D,L,B)
        
        for face_code, face_desc in face_order:
            colors = self.calibrate_face(face_code, face_desc)
            
            if colors is None:
                self.cap.release()
                cv2.destroyAllWindows()
                raise Exception("Calibration cancelled")
            
            # Store center sticker to build color-to-face mapping
            center_color = colors[4]  # Center is index 4 in 3x3 grid
            
            if center_color in center_map:
                print(f"⚠ Warning: Color {center_color} already mapped to face {center_map[center_color]}")
                print(f"  Now trying to map to {face_code}. This indicates a detection error!")
            
            center_map[center_color] = face_code
            all_colors.extend(colors)
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        
        # Build the raw string
        raw_string = ''.join(all_colors)
        print(f"\nRaw detected colors: {raw_string}")
        print(f"\nColor-to-Face mapping:")
        for color, face in sorted(center_map.items()):
            print(f"  {color} -> {face}")
        
        # Verify we have exactly 6 unique center colors
        if len(center_map) != 6:
            raise Exception(
                f"Error: Only detected {len(center_map)} unique center colors.\n"
                f"Expected 6 different colors. Check your calibration!"
            )
        
        # Translate from color codes (W,Y,R,O,B,G) to Kociemba face codes (U,R,F,D,L,B)
        kociemba_string = ''.join([center_map.get(c, '?') for c in raw_string])
        
        print(f"\nKociemba format string: {kociemba_string}")
        
        if '?' in kociemba_string:
            raise Exception("Error: Unknown color detected during translation!")
        
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
        - Only contains U, R, F, D, L, B
        - Exactly 9 of each letter
        """
        print("\n" + "="*60)
        print("VALIDATING CUBE STATE")
        print("="*60)
        
        if len(cube_string) != 54:
            raise ValueError(f"✗ Invalid length: {len(cube_string)} (expected 54)")
        
        required_faces = {'U', 'R', 'F', 'D', 'L', 'B'}
        face_counts = Counter(cube_string)
        
        # Check each face appears exactly 9 times
        print("\nFace count verification:")
        for face in required_faces:
            count = face_counts.get(face, 0)
            status = "✓" if count == 9 else "✗"
            print(f"  {status} Face {face}: {count}/9")
            if count != 9:
                raise ValueError(f"Face {face} appears {count} times (expected 9)")
        
        # Check no invalid characters
        invalid = set(cube_string) - required_faces
        if invalid:
            raise ValueError(f"✗ Invalid characters detected: {invalid}")
        
        print("\n✓ Cube validation passed - state is structurally valid")
        return True
    
    @staticmethod
    def solve(cube_string, max_depth=24):
        """
        Solve the cube using Kociemba two-phase algorithm.
        
        Args:
            cube_string: 54-char string with face notation (U,R,F,D,L,B)
            max_depth: Maximum search depth (default 24)
        
        Returns:
            solution: String of moves (e.g., "D2 R' L B U2 F")
        """
        print("\n" + "="*60)
        print("SOLVING CUBE")
        print("="*60)
        
        try:
            print(f"Running Kociemba algorithm (max depth: {max_depth})...")
            
            # Call kociemba solver
            # Signature: kociemba.solve(cubestring, patternstring="", max_depth=24)
            solution = kociemba.solve(cube_string, max_depth=max_depth)
            
            print(f"\n✓ Solution found!")
            print(f"  Moves: {solution}")
            print(f"  Total moves: {len(solution.split())}")
            
            return solution
            
        except ValueError as e:
            error_msg = str(e)
            print(f"\n✗ Kociemba solver error: {error_msg}")
            
            if "invalid" in error_msg.lower():
                raise Exception(
                    "\n" + "="*60 + "\n"
                    "INVALID CUBE CONFIGURATION\n" +
                    "="*60 + "\n"
                    "The cube state you detected is physically impossible.\n\n"
                    "Common causes:\n"
                    "  1. Color detection errors (wrong colors identified)\n"
                    "  2. Wrong face orientation during calibration\n"
                    "  3. Cube was disassembled and reassembled incorrectly\n"
                    "  4. Lighting issues causing misidentification\n\n"
                    "Solutions:\n"
                    "  - Recalibrate with better lighting\n"
                    "  - Ensure each face is clearly visible\n"
                    "  - Verify center colors are correct\n"
                    "  - Try adjusting HSV color ranges in code\n"
                )
            else:
                raise Exception(f"Solver failed: {error_msg}")
                
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")


# ============================================================================
# MOVE TRANSLATOR (for future Arduino integration)
# ============================================================================

class MoveTranslator:
    """
    Translates Kociemba notation to Fixed-Clockwise format.
    (Ready for future Arduino integration)
    """
    
    @staticmethod
    def translate_move(move):
        """
        Translate Kociemba notation to Fixed-Clockwise format.
        
        Examples:
          R   -> R1  (90° CW)
          R2  -> R2  (180° CW)
          R'  -> R3  (270° CW = 90° CCW)
        """
        move = move.strip()
        
        if not move:
            return None
        
        face = move[0]  # U, D, L, R, F, B
        
        if len(move) == 1:
            return f"{face}1"  # Standard 90° CW
        elif move[1] == '2':
            return f"{face}2"  # Double 180° CW
        elif move[1] == "'":
            return f"{face}3"  # Prime 270° CW (= 90° CCW)
        else:
            raise ValueError(f"Invalid move notation: {move}")
    
    @staticmethod
    def translate_solution(solution_string):
        """
        Translate entire solution to Fixed-Clockwise format.
        Returns list of translated moves.
        """
        moves = solution_string.split()
        translated = [MoveTranslator.translate_move(m) for m in moves]
        return translated


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main workflow (Phase 1: Detection + Solution only)
    1. Detect cube state with Computer Vision
    2. Validate cube configuration
    3. Solve with Kociemba algorithm
    4. Display solution (no hardware execution yet)
    """
    
    print("\n" + "="*60)
    print("RUBIK'S CUBE SOLVER")
    print("Phase 1: Computer Vision + Kociemba Solution")
    print("="*60)
    
    try:
        # Step 1: Computer Vision Calibration
        print("\n[STEP 1] Starting Computer Vision calibration...")
        detector = ColorDetector()
        cube_string = detector.calibrate_all_faces()
        
        # Step 2: Validate
        print("\n[STEP 2] Validating cube state...")
        CubeSolver.validate_cube_string(cube_string)
        
        # Step 3: Solve
        print("\n[STEP 3] Solving cube...")
        solution = CubeSolver.solve(cube_string, max_depth=24)
        
        # Step 4: Translate for future use
        print("\n[STEP 4] Translating to Fixed-Clockwise format...")
        translated = MoveTranslator.translate_solution(solution)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\nOriginal solution:  {solution}")
        print(f"Translated moves:   {' '.join(translated)}")
        print(f"Total moves:        {len(translated)}")
        
        print("\n" + "="*60)
        print("READY FOR ARDUINO INTEGRATION")
        print("="*60)
        print("Next steps:")
        print("  1. Connect Arduino to COM3")
        print("  2. Upload motor control sketch")
        print("  3. Run execution mode to send these moves")
        
        return solution, translated
        
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        return None, None
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR OCCURRED")
        print("="*60)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    solution, translated = main()
    
    if solution:
        print("\n✓ Program completed successfully")
        print("  Solution saved in variables: 'solution' and 'translated'")
    else:
        print("\n✗ Program did not complete successfully")