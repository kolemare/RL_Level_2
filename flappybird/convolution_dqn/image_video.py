import cv2
import numpy as np


class ImageVideo:
    # Static field to store images
    images = []
    training = False

    @staticmethod
    def add_image(pil_image):
        """
        Add a PIL image to the static `images` list after converting it to a NumPy array.

        Parameters:
        - pil_image (PIL.Image.Image): The PIL image to add.
        """
        ImageVideo.images.append(np.array(pil_image))

    @staticmethod
    def create_video(output_path: str):
        """
        Create an MP4 video from the static `images` field.

        Parameters:
        - output_path (str): The path to save the output video.
        """
        if not ImageVideo.images:
            raise ValueError("No images available to create a video. Add images to the `ImageVideo.images` list.")

        # Get the dimensions of the first image
        height, width, channels = ImageVideo.images[0].shape
        size = (width, height)

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        fps = 20
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        for img in ImageVideo.images:
            if img.shape[:2] != (height, width):
                raise ValueError("All images must have the same dimensions.")
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

        out.release()
        print(f"Video saved to {output_path}")
