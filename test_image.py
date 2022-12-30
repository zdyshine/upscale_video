"""
Copyright (c) 2022, David Lee
Author: David Lee
"""
import logging
import os
import sys
import cv2
from ncnn_vulkan import ncnn
import numpy as np
import math
import argparse
import subprocess


def transpose_tile(tile_output, output):
    output[
        tile_output[1] : tile_output[2], tile_output[3] : tile_output[4], :
    ] = tile_output[0][
        tile_output[5] : tile_output[6], tile_output[7] : tile_output[8], :
    ]


def process_tile(
    net,
    img,
    input_name,
    output_name,
    tile_size,
    scale,
    y,
    x,
    height,
    width,
):
    # extract tile from input image
    ofs_y = y * tile_size
    ofs_x = x * tile_size

    # input tile area on total image
    input_start_y = ofs_y
    input_end_y = min(ofs_y + tile_size, height)
    input_start_x = ofs_x
    input_end_x = min(ofs_x + tile_size, width)

    # calculate borders to help ai scale between tiles

    if input_start_y >= 10:
        b_start_y = -10
    else:
        b_start_y = 0

    if input_end_y <= height - 10:
        b_end_y = 10
    else:
        b_end_y = 0

    if input_start_x >= 10:
        b_start_x = -10
    else:
        b_start_x = 0

    if input_end_x <= width - 10:
        b_end_x = 10
    else:
        b_end_x = 0

    # input tile dimensions
    input_tile = img[
        input_start_y + b_start_y : input_end_y + b_end_y,
        input_start_x + b_start_x : input_end_x + b_end_x,
        :,
    ]
    input_tile = input_tile.copy()

    # Convert image to ncnn Mat

    mat_in = ncnn.Mat.from_pixels(
        input_tile,
        ncnn.Mat.PixelType.PIXEL_BGR,
        input_tile.shape[1],
        input_tile.shape[0],
    )
    mean_vals = []
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    # upscale tile
    try:
        # Make sure the input and output names match the param file
        ex = net.create_extractor()
        ex.input(input_name, mat_in)
        ret, mat_out = ex.extract(output_name)
        output_tile = np.array(mat_out)
    except RuntimeError as error:
        logging.error(error)
        sys.exit("upscale failed")

    # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
    output_tile = output_tile.transpose(1, 2, 0) * 255

    # scale area on total image
    input_start_y = input_start_y * scale
    b_start_y = b_start_y * scale
    input_end_y = input_end_y * scale
    input_start_x = input_start_x * scale
    b_start_x = b_start_x * scale
    input_end_x = input_end_x * scale

    # put tile into output image
    return (
        output_tile,
        input_start_y,
        input_end_y,
        input_start_x,
        input_end_x,
        -1 * b_start_y,
        input_end_y - input_start_y - b_start_y,
        -1 * b_start_x,
        input_end_x - input_start_x - b_start_x,
    )


def process_model(net, input_name, output_name, input_file, output_model_name):

    # Load image using opencv
    img = cv2.imread(input_file)

    mat_in = ncnn.Mat.from_pixels(
        img,
        ncnn.Mat.PixelType.PIXEL_BGR,
        img.shape[1],
        img.shape[0],
    )
    mean_vals = []
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    # Try/except block to catch out-of-memory error
    try:
        # Make sure the input and output names match the param file
        ex = net.create_extractor()
        ex.input(input_name, mat_in)
        ret, mat_out = ex.extract(output_name)
        out = np.array(mat_out)

        # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
        output = out.transpose(1, 2, 0) * 255

        # Save image using opencv
        cv2.imwrite(input_file + "." + output_model_name + ".png", output)
    except RuntimeError as error:
        logging.error(error)
        sys.exit("processing failed")

    logging.info("Processed " + output_model_name)


def process_image(
    input_file,
    output_file,
    scale,
    anime,
    denoise,
):

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    if denoise > 30:
        denoise = 30

    if denoise <= 0:
        denoise = None

    net = ncnn.Net()

    # Use vulkan compute
    net.opt.use_vulkan_compute = True

    model_path = os.path.realpath(__file__).split(os.sep)
    model_path = os.sep.join(model_path[:-1] + ["models"])

    logging.info("Processing File: " + input_file)

    if not output_file:
        output_file = input_file.split(".")
        output_file_ext = output_file[-1]
        output_file = ".".join(output_file[:-1] + [str(scale) + "x", output_file_ext])

    if anime:
        logging.info("Starting anime touchup...")
        net.load_param(os.path.join(model_path, "1x_BleedOut_Compact_300k_net_g.param"))
        net.load_model(os.path.join(model_path, "1x_BleedOut_Compact_300k_net_g.bin"))
        input_name = "input"
        output_name = "output"
        process_model(net, input_name, output_name, input_file, "anime")
        input_file = input_file + ".anime.png"

    if denoise:
        logging.info("Starting denoise touchup...")
        img = cv2.UMat(cv2.imread(input_file))
        input_file = input_file + ".denoise.png"
        output = cv2.fastNlMeansDenoisingColored(img, None, denoise, 10, 5, 9)
        cv2.imwrite(input_file, output)

    img = cv2.imread(input_file)
    net.load_param(os.path.join(model_path, str(scale) + "x_Compact_Pretrain.param"))
    net.load_model(os.path.join(model_path, str(scale) + "x_Compact_Pretrain.bin"))
    input_name = "input"
    output_name = "output"

    tile_size = 480

    height, width, batch = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (output_height, output_width, batch)

    # start with black image
    output = np.zeros(output_shape)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    for y in range(tiles_y):
        for x in range(tiles_x):
            tile_idx = y * tiles_x + x + 1
            logging.debug(f"\tProcessing Tile: {tile_idx}/{tiles_x * tiles_y}")

            output_tile = process_tile(
                net,
                img,
                input_name,
                output_name,
                tile_size,
                scale,
                y,
                x,
                height,
                width,
            )
            transpose_tile(output_tile, output)

    cv2.imwrite(output_file, output)

    logging.info("Completed")

    ncnn.destroy_gpu_instance()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test Image Upscaler")
    parser.add_argument("-i", "--input_file", required=True, help="input file.")
    parser.add_argument(
        "-o",
        "--output_file",
        help="optional output video file. Default is input_file + ('.2x.' or '.4x.')",
    )
    parser.add_argument(
        "-a",
        "--anime",
        action="store_true",
        help="Adds additional processing for anime videos to remove grain and color bleeding.",
    )
    parser.add_argument(
        "-n",
        "--denoise",
        type=int,
        default=0,
        help="Adds additional processing to reduce film grain. Denoise level 1 to 30. 3 = light / 10 = heavy.",
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=2, help="scale 2 or 4. Default is 2."
    )
    args = parser.parse_args()

    process_image(
        args.input_file,
        args.output_file,
        args.scale,
        args.anime,
        args.denoise,
    )
