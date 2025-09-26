"""
Unified Lane Detection System
============================
A comprehensive lane detection system that combines multiple advanced techniques
to work well in all situations including challenging conditions.

Author: Lane Detection System
Date: 2025
"""

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from collections import deque
import argparse
import os

class UnifiedLaneDetector:
    """
    Unified lane detection system optimized for stable, accurate lane detection
    without flickering or false lane detection.
    """
    
    def __init__(self, detection_mode='auto'):
        """
        Initialize the unified lane detector
        
        Args:
            detection_mode (str): 'easy', 'challenging', or 'auto'
        """
        self.detection_mode = detection_mode
        
        # History for temporal smoothing (shorter queues for more responsive detection)
        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)
        
        # Frame tracking
        self.frame_count = 0
        self.failed_detections = 0
        
        # Last known good lane positions for stability
        self.last_left_fit = None
        self.last_right_fit = None
        
        # Adaptive parameters based on mode
        self.params = self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters optimized for stable detection"""
        if self.detection_mode == 'easy':
            return {
                'canny_low': 50, 'canny_high': 150,
                'hough_threshold': 30, 'min_line_length': 50, 'max_line_gap': 50,
                'white_threshold': 200, 'yellow_s_threshold': 100
            }
        elif self.detection_mode == 'challenging':
            return {
                'canny_low': 40, 'canny_high': 120,
                'hough_threshold': 25, 'min_line_length': 40, 'max_line_gap': 30,
                'white_threshold': 180, 'yellow_s_threshold': 80
            }
        else:  # auto mode - balanced for stability
            return {
                'canny_low': 50, 'canny_high': 140,
                'hough_threshold': 30, 'min_line_length': 50, 'max_line_gap': 40,
                'white_threshold': 200, 'yellow_s_threshold': 90
            }
    
    def create_binary_mask(self, image):
        """
        Create stable binary mask focusing on clear lane detection
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Convert to HLS for better white/yellow lane separation
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        
        # White lane detection - high L channel values
        white_binary = np.zeros_like(l_channel)
        white_binary[l_channel >= self.params['white_threshold']] = 1
        
        # Yellow lane detection - specific H and S ranges
        yellow_binary = np.zeros_like(s_channel)
        yellow_binary[((h_channel >= 15) & (h_channel <= 35)) & 
                     (s_channel >= self.params['yellow_s_threshold'])] = 1
        
        # Combine white and yellow detection
        color_binary = np.zeros_like(gray)
        color_binary[(white_binary == 1) | (yellow_binary == 1)] = 1
        
        # Sobel X gradient for edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= self.params['canny_low']) & 
                    (scaled_sobel <= self.params['canny_high'])] = 1
        
        # Combine color and gradient detection
        combined_binary = np.zeros_like(gray)
        combined_binary[(color_binary == 1) | (sobel_binary == 1)] = 1
        
        return combined_binary
    
    def apply_region_of_interest(self, binary_image):
        """Apply stable region of interest mask"""
        rows, cols = binary_image.shape
        
        # Define stable trapezoid vertices
        vertices = np.array([
            [(cols * 0.1, rows),                    # Bottom left
             (cols * 0.45, rows * 0.6),            # Top left
             (cols * 0.55, rows * 0.6),            # Top right  
             (cols * 0.9, rows)]                   # Bottom right
        ], dtype=np.int32)
        
        mask = np.zeros_like(binary_image)
        cv2.fillPoly(mask, vertices, 1)
        
        return cv2.bitwise_and(binary_image, mask)
    
    def detect_lane_lines(self, binary_image):
        """
        Detect lane lines using Hough transform for stability
        """
        lines = cv2.HoughLinesP(
            binary_image,
            rho=2,
            theta=np.pi/180,
            threshold=self.params['hough_threshold'],
            minLineLength=self.params['min_line_length'],
            maxLineGap=self.params['max_line_gap']
        )
        
        if lines is None:
            return None, None
        
        # Separate lines into left and right based on slope
        left_lines = []
        right_lines = []
        rows, cols = binary_image.shape
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip nearly horizontal lines
            if abs(y2 - y1) < 10:
                continue
                
            # Calculate slope
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                
                # Left lane: negative slope, left side of image
                if slope < -0.5 and x1 < cols * 0.5:
                    left_lines.append(line[0])
                # Right lane: positive slope, right side of image  
                elif slope > 0.5 and x1 > cols * 0.5:
                    right_lines.append(line[0])
        
        # Average lines to get lane representations
        left_lane = self._average_lines(left_lines, rows)
        right_lane = self._average_lines(right_lines, rows)
        
        return left_lane, right_lane
    
    def _average_lines(self, lines, img_height):
        """Average multiple line segments into a single lane line"""
        if not lines or len(lines) == 0:
            return None
            
        # Collect all points
        x_coords = []
        y_coords = []
        
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 4:
            return None
        
        # Fit a line through all points
        try:
            poly = np.polyfit(y_coords, x_coords, 1)
            
            # Generate line endpoints
            y1 = img_height
            y2 = int(img_height * 0.6)
            x1 = int(poly[0] * y1 + poly[1])
            x2 = int(poly[0] * y2 + poly[1])
            
            return [x1, y1, x2, y2]
            
        except np.linalg.LinAlgError:
            return None
    
    def smooth_lane_detection(self, left_lane, right_lane):
        """
        Apply temporal smoothing for stable lane detection
        """
        # Use current detection or fall back to history
        if left_lane is not None:
            self.left_fit_history.append(left_lane)
            self.last_left_fit = left_lane
        
        if right_lane is not None:
            self.right_fit_history.append(right_lane) 
            self.last_right_fit = right_lane
        
        # Average recent detections for stability
        if len(self.left_fit_history) > 0:
            left_avg = np.mean(self.left_fit_history, axis=0)
            left_smoothed = left_avg.astype(int) if left_avg is not None else self.last_left_fit
        else:
            left_smoothed = self.last_left_fit
            
        if len(self.right_fit_history) > 0:
            right_avg = np.mean(self.right_fit_history, axis=0)
            right_smoothed = right_avg.astype(int) if right_avg is not None else self.last_right_fit
        else:
            right_smoothed = self.last_right_fit
            
        return left_smoothed, right_smoothed
    
    def draw_lanes(self, image, left_lane, right_lane):
        """
        Draw stable lane lines and lane area on the image
        """
        result = image.copy()
        
        if left_lane is None and right_lane is None:
            return result
        
        # Draw lane lines
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 10)  # Blue for left
        
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane  
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 10)  # Red for right
        
        # Fill lane area if we have both lanes
        if left_lane is not None and right_lane is not None:
            # Create polygon for lane area
            left_x1, left_y1, left_x2, left_y2 = left_lane
            right_x1, right_y1, right_x2, right_y2 = right_lane
            
            # Define lane area points
            lane_pts = np.array([
                [left_x1, left_y1],    # Left bottom
                [left_x2, left_y2],    # Left top  
                [right_x2, right_y2],  # Right top
                [right_x1, right_y1]   # Right bottom
            ], np.int32)
            
            # Draw semi-transparent green lane area
            overlay = result.copy()
            cv2.fillPoly(overlay, [lane_pts], (0, 255, 0))
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        
        return result
    
    def process_frame(self, image):
        """
        Main processing pipeline - stable lane detection without flickering
        """
        self.frame_count += 1
        
        try:
            # Step 1: Create binary mask for lane detection
            binary_mask = self.create_binary_mask(image)
            
            # Step 2: Apply region of interest
            roi_binary = self.apply_region_of_interest(binary_mask)
            
            # Step 3: Detect lane lines using Hough transform
            left_lane, right_lane = self.detect_lane_lines(roi_binary)
            
            # Step 4: Apply temporal smoothing for stability
            left_smooth, right_smooth = self.smooth_lane_detection(left_lane, right_lane)
            
            # Step 5: Draw lanes on the original image
            result = self.draw_lanes(image, left_smooth, right_smooth)
            
            # Track successful detections
            if left_lane is not None or right_lane is not None:
                self.failed_detections = 0
            else:
                self.failed_detections += 1
            
            return result
            
        except Exception as e:
            print(f"Warning: Frame {self.frame_count} processing failed: {e}")
            self.failed_detections += 1
            return image  # Return original image on failure

def process_video(input_path, output_path, detection_mode='auto'):
    """
    Process a video file with lane detection
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        detection_mode (str): 'easy', 'challenging', or 'auto'
    """
    print(f"Processing video: {input_path}")
    print(f"Detection mode: {detection_mode}")
    print(f"Output will be saved to: {output_path}")
    
    # Initialize detector
    detector = UnifiedLaneDetector(detection_mode=detection_mode)
    
    # Process video
    clip = VideoFileClip(input_path)
    processed_clip = clip.fl_image(detector.process_frame)
    processed_clip.write_videofile(output_path, audio=False)
    
    print(f"Processing complete! Frames processed: {detector.frame_count}")
    print(f"Failed detections: {detector.failed_detections}")

def process_image(input_path, output_path, detection_mode='auto'):
    """
    Process a single image with lane detection
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to output image
        detection_mode (str): 'easy', 'challenging', or 'auto'
    """
    print(f"Processing image: {input_path}")
    print(f"Detection mode: {detection_mode}")
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image {input_path}")
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize detector and process
    detector = UnifiedLaneDetector(detection_mode=detection_mode)
    result = detector.process_frame(image)
    
    # Convert back to BGR and save
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result)
    
    print(f"Image processing complete! Saved to: {output_path}")

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description='Unified Lane Detection System')
    parser.add_argument('input', help='Input video or image file path')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-m', '--mode', choices=['easy', 'challenging', 'auto'], 
                        default='auto', help='Detection mode (default: auto)')
    parser.add_argument('--type', choices=['video', 'image', 'auto'], 
                        default='auto', help='Input type (default: auto)')
    
    args = parser.parse_args()
    
    # Auto-detect input type if not specified
    if args.type == 'auto':
        if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_type = 'video'
        elif args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_type = 'image'
        else:
            print("Error: Could not detect input type. Please specify --type")
            return
    else:
        input_type = args.type
    
    # Generate output path if not provided
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_lane_detected{ext}"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Process based on type
    if input_type == 'video':
        process_video(args.input, args.output, args.mode)
    else:
        process_image(args.input, args.output, args.mode)

# Predefined configurations for common scenarios
def process_challenge_video():
    """Process the challenge video with optimal settings"""
    input_video = r"C:\Users\dasul\Downloads\Lane Line Detection System\test_vedios\challenge.mp4"
    output_video = "outputs/challenge_unified_output.mp4"
    process_video(input_video, output_video, detection_mode='challenging')

def process_easy_video():
    """Process an easy video (solidWhiteRight) with optimal settings"""
    input_video = r"C:\Users\dasul\Downloads\Lane Line Detection System\test_vedios\solidWhiteRight.mp4"
    output_video = "outputs/solidwhite_unified_output.mp4"
    process_video(input_video, output_video, detection_mode='easy')

def process_test_image():
    """Process a test image with optimal settings"""
    input_image = "test_image/solidWhiteRight.jpg"
    output_image = "outputs/test_unified_output.jpg"
    process_image(input_image, output_image, detection_mode='auto')

if __name__ == "__main__":
    # If run directly, process the challenge video as default
    if len(os.sys.argv) == 1:
        print("Running default: Challenge video processing...")
        process_challenge_video()
    else:
        main()