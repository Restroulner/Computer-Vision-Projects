import cv2
import numpy as np

def apply_gaussian_blur(image_path, kernel_size=(5, 5), sigmaX=0):
    """
    Applies a Gaussian blur filter to an image.

    Args:
        image_path (str): Path to the input image.
        kernel_size (tuple): Gaussian kernel size. Width and height should be odd and positive.
        sigmaX (int): Gaussian kernel standard deviation in X direction.

    Returns:
        numpy.ndarray: Blurred image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    blurred_img = cv2.GaussianBlur(img, kernel_size, sigmaX)
    return blurred_img

def apply_sobel_edge_detection(image_path, ksize=3):
    """
    Applies Sobel edge detection to an image.

    Args:
        image_path (str): Path to the input image.
        ksize (int): Size of the extended Sobel kernel; it must be 1, 3, 5, or 7.

    Returns:
        numpy.ndarray: Image with detected edges.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gradient_magnitude

if __name__ == "__main__":
    # Create a dummy image for demonstration
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(dummy_image, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite("dummy_image.png", dummy_image)

    # Example usage of Gaussian blur
    blurred_image = apply_gaussian_blur("dummy_image.png")
    if blurred_image is not None:
        cv2.imwrite("blurred_dummy_image.png", blurred_image)
        print("Blurred image saved as blurred_dummy_image.png")

    # Example usage of Sobel edge detection
    edges_image = apply_sobel_edge_detection("dummy_image.png")
    if edges_image is not None:
        cv2.imwrite("edges_dummy_image.png", edges_image)
        print("Edges image saved as edges_dummy_image.png")

    # Clean up dummy image
    import os
    os.remove("dummy_image.png")
    os.remove("blurred_dummy_image.png")
    os.remove("edges_dummy_image.png")
