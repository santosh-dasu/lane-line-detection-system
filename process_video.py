from unified_lane_detection import process_video

if __name__ == "__main__":
    # Use unified system for video processing
    input_video = r"C:\Users\dasul\Downloads\Lane Line Detection System\test_vedios\solidWhiteRight.mp4"
    output_video = "outputs/solidwhite_unified_output.mp4"
    
    # Process with auto-adaptive mode
    process_video(input_video, output_video, detection_mode='auto')
