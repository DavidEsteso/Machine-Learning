import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def convolve2d(input_matrix, kernel_matrix, padding='valid', stride=(1, 1)):
    # Get kernel dimensions
    kernel_height, kernel_width = kernel_matrix.shape
    
    # Apply padding if 'same' is specified
    if padding == 'same':
        padding_height = (kernel_height - 1) // 2
        padding_width = (kernel_width - 1) // 2
        padded_input_matrix = np.pad(input_matrix, 
                                     ((padding_height, padding_height), (padding_width, padding_width)), 
                                     mode='constant')
    elif padding == 'valid':
        padded_input_matrix = input_matrix  # No padding for 'valid'
    else:
        raise ValueError("Invalid padding type. Use 'same' or 'valid'.")

    # Get padded input matrix dimensions
    padded_input_height, padded_input_width = padded_input_matrix.shape

    # Get stride values
    stride_vertical, stride_horizontal = stride

    # Calculate output matrix dimensions
    output_height = (padded_input_height - kernel_height) // stride_vertical + 1
    output_width = (padded_input_width - kernel_width) // stride_horizontal + 1

    # Initialize output matrix
    output_matrix = np.zeros((output_height, output_width))

    # Perform convolution
    for output_row_index in range(output_height):
        for output_column_index in range(output_width):
            # Extract region of the input corresponding to the kernel
            row_start = output_row_index * stride_vertical
            row_end = row_start + kernel_height
            col_start = output_column_index * stride_horizontal
            col_end = col_start + kernel_width
            input_region = padded_input_matrix[row_start:row_end, col_start:col_end]
            
            # Convolve (element-wise multiplication and sum)
            output_matrix[output_row_index, output_column_index] = np.sum(input_region * kernel_matrix)
    
    return output_matrix 

def process_image(image_path, kernel, id):
    # Load the image and convert it to an RGB array
    im = Image.open(image_path)
    rgb = np.array(im.convert('RGB'))

    # Extract the red channel (R) from the RGB image
    r_channel = rgb[:, :, 0]

    # Save the red channel as a separate image
    if id == "kernel2":
        plt.imshow(r_channel, cmap='gray')
        plt.axis('off')

        plt.savefig(f'../output/RED_{id}.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()

    # Apply the 2D convolution on the red channel
    output_array = convolve2d(r_channel, kernel)

    # Clip values to the range [0, 255] and convert the result back to an image
    output_image = Image.fromarray(np.uint8(np.clip(output_array, 0, 255)))

    # Save the processed image to the output folder with an identifier
    output_image.save(f'../output/RGB_{id}.jpg')