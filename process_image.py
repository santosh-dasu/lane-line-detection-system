from unified_lane_detection import process_image

if __name__ == "__main__":
    # Use unified system for image processing
    input_image = "test_image/solidWhiteRight.jpg"
    output_image = "outputs/test_unified_output.jpg"
    
    # Process with auto-adaptive mode
    process_image(input_image, output_image, detection_mode='auto')
