# Import libraries
import numpy as np
import json
import math
import pygame
import sys

# ============================================================================================================================================== #
# Rectangle Class Definition
class Rectangle:

    # Initialize a rectangle using two opposite diagonal points
    def __init__(self, p1, p2):

        x1, y1 = p1
        x2, y2 = p2

        # Store the four corners of the rectangle in homogeneous coordinates
        self.original_corners = np.array([[x1, y1, 1],
                                          [x2, y1, 1],
                                          [x2, y2, 1],
                                          [x1, y2, 1]],
                                          dtype = np.float32)
        
        # Initialize with identity transform
        self.transform = np.eye(3)

    # Apply transformation matrix and return the updated 2D corner coordinates
    def get_transformed_corners(self):

        return (self.original_corners @ self.transform.T)[:, :2]

    # Draw the current rectangle on a Pygame surface using transformed corners
    def draw(self, surface, color = (0, 128, 255)):

        pts = self.get_transformed_corners().astype(int)

        # Draw lines between consecutive corners
        for i in range(4):
            pygame.draw.line(surface, color, pts[i], pts[(i + 1) % 4], 2)

    # Check if the given point is within a certain distance to any corner
    def is_point_near(self, x, y, threshold = 10):

        corners = self.get_transformed_corners()
        for cx, cy in corners:
            if abs(cx - x) < threshold and abs(cy - y) < threshold:
                return True
            
        return False

    # Convert rectangle's current state to a dictionary
    def to_dict(self):

        return {"original_corners": self.original_corners[:, :2].tolist(),
                "transform": self.transform.tolist()}

    @staticmethod

    # Load a rectangle instance from saved dictionary data
    def from_dict(data):

        # Create dummy rectangle
        rect = Rectangle((0, 0), (0, 0))

        # Restore corners with added homogeneous coordinate
        rect.original_corners = np.array([[*pt, 1] for pt in data["original_corners"]], dtype = np.float32)
        rect.transform = np.array(data["transform"], dtype = np.float32)

        return rect

    # Apply pure translation to the rectangle by dx and dy
    def apply_translation(self, dx, dy):

        T = np.array([[1, 0, dx],
                      [0, 1, dy],
                      [0, 0, 1]],
                      dtype = np.float32)
        
        self.transform = T @ self.transform

    # Apply rigid transformation
    def apply_rigid_transform(self, origin, new_pos):

        center = np.mean(self.get_transformed_corners(), axis = 0)
        v1 = np.array(origin) - center
        v2 = np.array(new_pos) - center

        theta = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        R = np.array([[cos_t, -sin_t, 0],
                      [sin_t, cos_t, 0],
                      [0, 0, 1]],
                      dtype = np.float32)
        
        T1 = np.array([[1, 0, -center[0]],
                       [0, 1, -center[1]],
                       [0, 0, 1]],
                       dtype = np.float32)
        
        T2 = np.array([[1, 0, center[0]],
                       [0, 1, center[1]],
                       [0, 0, 1]],
                       dtype = np.float32)
        
        self.transform = T2 @ R @ T1 @ self.transform

    # Apply similarity transformation
    def apply_similarity_transform(self, origin, new_pos):

        center = np.mean(self.get_transformed_corners(), axis = 0)
        v1 = np.array(origin) - center
        v2 = np.array(new_pos) - center

        theta = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
        scale = len2 / len1 if len1 != 0 else 1

        cos_t = math.cos(theta) * scale
        sin_t = math.sin(theta) * scale

        RS = np.array([[cos_t, -sin_t, 0],
                       [sin_t, cos_t, 0],
                       [0, 0, 1]],
                       dtype = np.float32)
        
        T1 = np.array([[1, 0, -center[0]],
                       [0, 1, -center[1]],
                       [0, 0, 1]],
                       dtype = np.float32)
        
        T2 = np.array([[1, 0, center[0]],
                       [0, 1, center[1]],
                       [0, 0, 1]],
                       dtype = np.float32)
        
        self.transform = T2 @ RS @ T1 @ self.transform

    # Apply affine transformation using one manipulated corner and two of its neighbors
    def apply_affine_transform(self, selected_idx, new_pos):

        src = self.get_transformed_corners()

        # Get three points
        idx = [selected_idx, (selected_idx + 1) % 4, (selected_idx + 3) % 4]
        src_pts = src[idx]
        dst_pts = np.array(src_pts)
        dst_pts[0] = new_pos

        A = self.compute_affine_matrix(src_pts, dst_pts)
        if A is not None:
            T = np.vstack([A, [0, 0, 1]])
            self.transform = T @ self.transform

    # Estimate affine matrix using 3 point correspondences via least squares
    def compute_affine_matrix(self, src, dst):

        A, b = [], []

        for (x, y), (xp, yp) in zip(src, dst):
            A.append([x, y, 1, 0, 0, 0])
            A.append([0, 0, 0, x, y, 1])
            b += [xp, yp]

        A, b = np.array(A), np.array(b)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond = None)

        return x.reshape(2, 3)

    # Apply perspective transformation by moving one corner
    def apply_perspective_transform(self, idx, new_pos):

        src = self.get_transformed_corners()
        dst = np.array(src)
        dst[idx] = new_pos

        H = self.compute_homography(src, dst)
        if H is not None:
            self.transform = H @ self.transform

    # Compute 3x3 homography matrix
    def compute_homography(self, src, dst):

        A = []

        for (x, y), (xp, yp) in zip(src, dst):
            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

        A = np.array(A)

        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)

        return H / H[2, 2]

# ============================================================================================================================================== #
# Main Application Implementation
pygame.init()

# Set up canvas dimensions and display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Transform Editor")
font = pygame.font.SysFont("Arial", 20)

# Clock for managing frame rate
clock = pygame.time.Clock()

# Application state variables
rectangles = []
current_rect = None
dragging = False
selected_rect = None
selected_corner_idx = None
drag_start = None

# Supported transformation modes and their keyboard bindings
MODES = ["Translation", "Rigid", "Similarity", "Affine", "Perspective"]
mode_keys = {pygame.K_t: "Translation",
             pygame.K_r: "Rigid",
             pygame.K_m: "Similarity",
             pygame.K_a: "Affine",
             pygame.K_p: "Perspective"}
current_mode = "Translation"

# Save all rectangles to a JSON file
def save_rectangles(filename = "rectangles.json"):
    
    data = [r.to_dict() for r in rectangles]

    with open(filename, "w") as f:
        json.dump(data, f, indent = 2)

# Load rectangles from JSON file
def load_rectangles(filename = "rectangles.json"):

    global rectangles

    with open(filename, "r") as f:
        data = json.load(f)

    rectangles = [Rectangle.from_dict(d) for d in data]

# Create a pygame.Rect between two screen points
def get_rect_from_points(p1, p2):

    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    w = abs(p2[0] - p1[0])
    h = abs(p2[1] - p1[1])

    return pygame.Rect(x, y, w, h)

# Main event loop
while True:

    # Clear screen with white background
    screen.fill((255, 255, 255))

    keys = pygame.key.get_pressed()
    shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

    for event in pygame.event.get():

        # Quit the application
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Handle keyboard inputs
        elif event.type == pygame.KEYDOWN:
            if event.key in mode_keys:
                current_mode = mode_keys[event.key]
            elif event.key == pygame.K_s:
                save_rectangles()
            elif event.key == pygame.K_l:
                load_rectangles()

        # Mouse button pressed
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = event.pos

                # Start rubber-band rectangle creation
                if shift_held:
                    start_pos = (x, y)
                    dragging = True
                
                # Try to select a corner from existing rectangles
                else:
                    for rect in reversed(rectangles):
                        corners = rect.get_transformed_corners()
                        for i, (cx, cy) in enumerate(corners):
                            if abs(cx - x) < 10 and abs(cy - y) < 10:
                                selected_rect = rect
                                selected_corner_idx = i
                                drag_start = np.array([x, y])
                                break
                        if selected_rect:
                            break

        # Mouse button released
        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging:
                end_pos = pygame.mouse.get_pos()
                rect = Rectangle(start_pos, end_pos)
                rectangles.append(rect)
                dragging = False

            # Reset selection state
            selected_rect = None
            drag_start = None

        # Mouse is moving while pressed
        elif event.type == pygame.MOUSEMOTION:
            if selected_rect and drag_start is not None:
                x, y = event.pos

                # Apply transformation according to the current mode
                if current_mode == "Translation":
                    dx, dy = x - drag_start[0], y - drag_start[1]
                    selected_rect.apply_translation(dx, dy)
                elif current_mode == "Rigid":
                    selected_rect.apply_rigid_transform(drag_start, (x, y))
                elif current_mode == "Similarity":
                    selected_rect.apply_similarity_transform(drag_start, (x, y))
                elif current_mode == "Affine":
                    selected_rect.apply_affine_transform(selected_corner_idx, (x, y))
                elif current_mode == "Perspective":
                    selected_rect.apply_perspective_transform(selected_corner_idx, (x, y))
                
                # Update drag_start for continuous transformation
                drag_start = np.array([x, y])

    # Draw all rectangles
    for rect in rectangles:
        color = (255, 0, 0) if rect == selected_rect else (0, 128, 255)
        rect.draw(screen, color)

    # Draw a preview rectangle if user is in drawing mode
    if dragging and shift_held:
        cur_pos = pygame.mouse.get_pos()
        temp_rect = get_rect_from_points(start_pos, cur_pos)
        pygame.draw.rect(screen, (0, 0, 0), temp_rect, 1)

    # Display mode info and instructions at bottom
    text = font.render(f"Mode: {current_mode}  |  [T/R/M/A/P]  Save: S  Load: L", True, (0, 0, 0))
    screen.blit(text, (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(60)
