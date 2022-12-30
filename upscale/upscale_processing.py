"""
Copyright (c) 2022, David Lee
Author: David Lee
"""
import logging
import os
import subprocess
import tempfile
import shutil
import sys
import cv2
from ncnn_vulkan import ncnn
import numpy as np
import math
from wakepy import set_keepawake, unset_keepawake
from multiprocessing import Pool


def get_frames_per_sec(ffmpeg, input_file):
    logging.info("Getting frame timestamps from " + str(input_file))
    if not os.path.exists("timecode.txt"):
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-hwaccel",
                "auto",
                "-i",
                input_file,
                "-f",
                "mkvtimestamp_v2",
                "timecode.txt",
            ]
        )
    with open("timecode.txt") as f:
        skip_header_comment_line = next(f)  ## timecode format v2
        lines = f.read().rstrip()

    frame_timestamps = [int(timestamp) for timestamp in lines.split("\n")]
    frames_per_sec = len(frame_timestamps) / frame_timestamps[-1] * 1000
    return frames_per_sec, len(frame_timestamps)


def get_crop_detect(ffmpeg, input_file, interval_check):
    logging.info("Getting crop_filter from " + str(input_file))
    width = []
    height = []

    for i in range(19):
        results = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-ss",
                str(interval_check * (i + 1)),
                "-i",
                input_file,
                "-vframes",
                "10",
                "-vf",
                "cropdetect",
                "-f",
                "null",
                "-",
            ],
            shell=True,
            capture_output=True,
            text=True,
        )
        results = str(results).split(" ")
        for result_str in results:
            if result_str.startswith("crop="):
                crop = result_str.split("""\\n""")[0]
                logging.info("Crop Detected: " + crop)
                n_w, n_h, n_x, n_y = crop[5:].split(":")
                width.append((n_w, n_x))
                height.append((n_h, n_y))
                break

    if width and height:
        width = max(set(width), key=width.count)
        height = max(set(height), key=height.count)

        crop = "crop=" + ":".join([width[0], height[0], width[1], height[1]])
        return crop

    return None


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
        ncnn.destroy_gpu_instance()
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


def process_denoise(frame, denoise, input_model_name):
    img_file = str(frame + 1) + "." + input_model_name + ".png"

    if not os.path.exists(img_file):
        return img_file + " not found - skipping denoise"

    img = cv2.UMat(cv2.imread(img_file))

    output = cv2.fastNlMeansDenoisingColored(img, None, denoise, 10, 5, 9)

    cv2.imwrite(str(frame + 1) + ".denoise.png", output)

    os.remove(img_file)

    return "Processed Frame: denoise " + str(frame)


def process_model(
    frame,
    input_model_name,
    output_model_name,
    net,
    input_name,
    output_name,
):

    if not os.path.exists(str(frame) + "." + input_model_name + ".png"):
        return

    # Load image using opencv
    img = cv2.imread(str(frame) + "." + input_model_name + ".png")

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
        cv2.imwrite(str(frame) + "." + output_model_name + ".png", output)
    except RuntimeError as error:
        logging.error(error)
        ncnn.destroy_gpu_instance()
        sys.exit("processing failed")

    os.remove(str(frame) + "." + input_model_name + ".png")

    logging.info("Processed Frame: " + output_model_name + " " + str(frame))


def upscale_frames(
    net,
    input_model_name,
    ffmpeg,
    ffmpeg_encoder,
    frame_batch,
    start_frame,
    end_frame,
    frames_per_sec,
    scale,
    input_name,
    output_name,
):

    logging.info(
        "Processing Batch: "
        + str(frame_batch)
        + " : Number of frames: "
        + str(1 + end_frame - start_frame)
    )

    if os.path.exists(str(frame_batch) + ".mkv"):
        return

    frames_upscaled = 0

    ## upscale frames
    for frame in range(start_frame, end_frame + 1):

        if os.path.exists(str(frame) + ".png"):
            frames_upscaled += 1
            continue

        # Load image using opencv
        img = cv2.imread(str(frame) + "." + input_model_name + ".png")

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

        cv2.imwrite(str(frame) + ".png", output)

        os.remove(str(frame) + "." + input_model_name + ".png")

        frames_upscaled += 1
        logging.info(
            "Processing Batch: "
            + str(frame_batch)
            + " : Upscaled "
            + str(frames_upscaled)
            + "/"
            + str(1 + end_frame - start_frame)
        )

    ## run ffmpeg to merge frames
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-hwaccel",
            "auto",
            "-r",
            str(frames_per_sec),
            "-f",
            "image2",
            "-start_number",
            str(start_frame),
            "-i",
            "%d.png",
            "-vcodec",
            ffmpeg_encoder,
            "-frames:v",
            str(1 + end_frame - start_frame),
            "-pix_fmt",
            "p010le",
            "-global_quality",
            "20",
            str(frame_batch) + ".mkv",
        ]
    )

    logging.info("Batch merged into " + str(frame_batch) + ".mkv")
    logging.info(str(end_frame) + " total frames upscaled")

    ## delete converted png files
    for i in range(start_frame, end_frame + 1):
        os.remove(str(i) + ".png")


def merge_mkvs(ffmpeg, frame_batches, output_file, log_dir):
    logging.info("Merging Fragments into " + output_file)
    with open("merge_list.txt", "w") as f:
        for i in range(frame_batches):
            f.write("file " + str(i + 1) + ".mkv\n")

    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "merge_list.txt",
            "-c",
            "copy",
            output_file,
        ],
        capture_output=True,
    )

    if log_dir:
        output_file = output_file.split(os.sep)[-1][:-4]
        with open(os.path.join(log_dir, output_file + ".log"), "wb") as f:
            f.write(result.stdout)
            f.write(result.stderr)


def process_file(
    input_file,
    output_file,
    ffmpeg,
    ffmpeg_encoder,
    scale,
    temp_dir,
    batch_size,
    resume_processing,
    extract_only,
    anime,
    denoise,
    log_level,
    log_dir,
):
    """
    Upscale video file from 2k to 8k
    :param input_file:
    :param output_file:
    :param ffmpeg:
    :param ffmpeg_encoder:
    :param scale:
    :param temp_dir:
    :param batch_size:
    :param resume_processing:
    :param extract_only:
    :param anime:
    :param denoise:
    :param log_level:
    :param log_dir
    """

    if not log_level:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    if denoise:
        if > 30:
            denoise = 30
        if denoise <= 0:
            denoise = None

    if scale not in [2, 4]:
        ncnn.destroy_gpu_instance()
        sys.exit("Scale must be 2 or 4")

    set_keepawake(keep_screen_awake=False)

    net = ncnn.Net()

    # Use vulkan compute
    net.opt.use_vulkan_compute = True

    model_path = os.path.realpath(__file__).split(os.sep)
    model_path = os.sep.join(model_path[:-2] + ["models"])

    logging.info("Processing File: " + input_file)

    ## Create temp directory
    if not temp_dir:
        temp_dir = tempfile.gettempdir()

    if not output_file:
        output_file = input_file.split(".")
        output_file_ext = output_file[-1]
        output_file = ".".join(output_file[:-1] + [str(scale) + "x", output_file_ext])

    temp_dir = os.path.join(temp_dir, "upscale_video")
    if os.path.exists(temp_dir):
        if not resume_processing:
            shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)
    else:
        os.mkdir(temp_dir)

    ## change working directory to temp directory
    cwd_dir = os.getcwd()
    os.chdir(temp_dir)

    ## get fps
    frames_per_sec, frames_count = get_frames_per_sec(ffmpeg, input_file)
    logging.info("Number of frames: " + str(frames_count))
    logging.info("Frames per second: " + str(frames_per_sec))
    ## calculate frames per minute
    frames_per_batch = int(frames_per_sec * 60) * batch_size

    crop_detect = get_crop_detect(
        ffmpeg, input_file, int(frames_count / frames_per_sec / 20)
    )

    cmds = [
        ffmpeg,
        "-hide_banner",
        "-hwaccel",
        "auto",
        "-i",
        input_file,
        "-pix_fmt",
        "rgb24",
    ]

    if crop_detect:
        logging.info("Final Crop: " + crop_detect)
        cmds.append("-vf")
        cmds.append(crop_detect)

    cmds.append("%d.extract.png")

    ## Extract frames to temp dir. Need 300 gigs for a 2 hour movie
    logging.info("Starting Frame Extracting..")

    if (
        not os.path.exists(str(frames_count) + ".extract.png")
        and not os.path.exists(str(frames_count) + ".anime.png")
        and not os.path.exists(str(frames_count) + ".denoise.png")
    ):
        subprocess.run(cmds)

    if extract_only:
        ncnn.destroy_gpu_instance()
        sys.exit("Extract Only - Frame Extracting Completed")

    input_model_name = "extract"

    if anime:
        logging.info("Starting anime touchup...")
        net.load_param(os.path.join(model_path, "1x_BleedOut_Compact_300k_net_g.param"))
        net.load_model(os.path.join(model_path, "1x_BleedOut_Compact_300k_net_g.bin"))
        input_name = "input"
        output_name = "output"

        for frame in range(frames_count):
            process_model(
                frame + 1,
                input_model_name,
                "anime",
                net,
                input_name,
                output_name,
            )

        input_model_name = "anime"

    if denoise:
        logging.info("Starting denoise touchup...")
        pool = Pool()

        for frame in range(frames_count):
            pool.apply_async(process_denoise, args=(frame, denoise, input_model_name), callback=logging.info)

        pool.close()
        pool.join()

        input_model_name = "denoise"

    logging.info("Starting upscale processing...")

    # Load model param and bin. Make sure input and output names match what is in the .param file
    net.load_param(os.path.join(model_path, str(scale) + "x_Compact_Pretrain.param"))
    net.load_model(os.path.join(model_path, str(scale) + "x_Compact_Pretrain.bin"))
    input_name = "input"
    output_name = "output"

    frame_batch = 1

    while frame_batch * frames_per_batch <= frames_count:
        start_frame = 1 + (frame_batch - 1) * frames_per_batch

        upscale_frames(
            net,
            input_model_name,
            ffmpeg,
            ffmpeg_encoder,
            frame_batch,
            start_frame,
            frame_batch * frames_per_batch,
            frames_per_sec,
            scale,
            input_name,
            output_name,
        )
        frame_batch += 1

    ## process remaining frames
    start_frame = 1 + (frame_batch - 1) * frames_per_batch
    if frames_count >= start_frame:
        upscale_frames(
            net,
            input_model_name,
            ffmpeg,
            ffmpeg_encoder,
            frame_batch,
            start_frame,
            frames_count,
            frames_per_sec,
            scale,
            input_name,
            output_name,
        )

    ## merge video files into a single video file
    merge_mkvs(ffmpeg, frame_batch, output_file, log_dir)

    logging.info("Upscale finished for " + output_file)

    os.chdir(cwd_dir)

    if input("Is video file ok? (y/n)") == "y":
        logging.info("Cleaning up temp directory")
        shutil.rmtree(temp_dir)
    else:
        logging.info("Exiting without removing " + temp_dir)
        logging.info("Fix any issue and rerun upscale_video.py with -r parameter")

    ncnn.destroy_gpu_instance()
    unset_keepawake()
