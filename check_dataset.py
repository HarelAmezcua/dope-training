import torch
import sys
from src.auxiliar import get_DataLoaders
import src.custom_transform as ct
import src.args_parser as ar
import src.auxiliar as aux

import os
import random
import datetime

import multiprocessing as mp
import time

import matplotlib.pyplot as plt

import numpy as np

import cv2

from PIL import ImageDraw, Image

class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)
        self.width = im.size[0]

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_rectangle(self, point1, point2, line_color=(0, 255, 0), line_width=2):
        self.draw.rectangle([point1, point2], outline=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius,
            ]
            self.draw.ellipse(xy, fill=point_color, outline=point_color)

    def draw_text(self, point, text, text_color):
        """Draws text on image"""
        if point is not None:
            self.draw.text(point, text, fill=text_color)

    def draw_cube(self, points, color=(0, 255, 0)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """
        line_order = [[0, 1], [1, 2], [3, 2], [3, 0],  # front
                      [4, 5], [6, 5], [6, 7], [4, 7], # back
                      [0, 4], [7, 3], [5, 1], [2, 6], # sides
                      [0, 5], [1,4]]                  # x on top

        for l in line_order:
            self.draw_line(points[l[0]], points[l[1]], color, line_width=2)
        # Draw center
        self.draw_dot(points[8], point_color=color, point_radius=6)

        for i in range(9):
            self.draw_text(points[i], str(i), (255, 0, 0))



def main():
    full_path = os.getcwd()
    sys.path.append(full_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse arguments
    opt = ar.parse_args(full_path, False)
    print(opt.outf)

    # set the manual seed.
    random.seed(opt.manualseed)
    torch.manual_seed(opt.manualseed)
    torch.cuda.manual_seed_all(opt.manualseed)
    # Create output folder and files
    aux.create_output_folder(opt)
    print ("start:" , datetime.datetime.now().time())

    # Initialize the image transforms
    transform, preprocessing_transform, mean, std  = ct.get_transform()

    # Get the DataLoaders
    train_dataset, test_dataset, trainingdata, testingdata = aux.get_DataLoaders(opt, preprocessing_transform, transform)


    for batch_idx in range(10):
        iterator = iter(trainingdata)
        data = next(iterator)

        # Visualize the first 8 images in the batch in a 2x4 grid
        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
        for i in range(8):
            image = data['image'][i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            image = image * std + mean  # Denormalize the image

            # Convert numpy array to PIL.Image
            image_pil = Image.fromarray((image * 255).astype(np.uint8))  # Scale to 0-255 and convert to uint8

            draw = Draw(image_pil)  # Pass the PIL.Image object to Draw
            keypoints = data['keypoints'][i].squeeze().cpu().numpy()

            # Ensure keypoints is a 2D array with shape (N, 2)
            if keypoints.ndim == 1:
                keypoints = keypoints.reshape(-1, 2)

            # Convert keypoints to a list of tuples with integer values
            keypoints = [tuple(map(int, point)) for point in keypoints]

            # Validate keypoints shape and type
            if len(keypoints) < 9:
                print(f"Invalid keypoints: expected at least 9 points, got {len(keypoints)}")
            else:
                if not torch.all(data['has_points_belief'][i] == 0):
                    draw.draw_cube(keypoints, color=(255, 0, 0))
                else:
                    print("Warning: keypoints is a tensor of bools, skipping drawing.")
            
            axes[i].imshow(image_pil)
            axes[i].axis('off')
            print(data['keypoints'][i].dtype)
        plt.show()

        print(f"Batch {batch_idx + 1}")
        print(data['image'].shape)
        print(data['translations'].shape)
        print(data['rotations'].shape)
        print(data['has_points_belief'].shape)
        print(data['keypoints'].shape)        

        time.sleep(10)  # Wait for 10 seconds
        plt.close(fig)  # Close the matplotlib window

if __name__ == "__main__":
    mp.freeze_support()
    main()