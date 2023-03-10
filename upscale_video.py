"""
Copyright (c) 2022, David Lee
Author: David Lee
"""
import argparse

from upscale.upscale_processing import process_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upscale Video 2x or 4x")
    parser.add_argument("-i", "--input_file", required=True, help="Input file.")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Optional output video file location. Default is input_file + ('.2x.' or '.4x.')",
    )
    parser.add_argument("-f", "--ffmpeg", required=True, help="Location of ffmpeg.")
    parser.add_argument(
        "-e",
        "--ffmpeg_encoder",
        default="av1_qsv",
        help="ffmpeg encoder for mkv file. Default is av1_qsv.",
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
        help="Adds additional processing to remove film grain. Denoise level 1 to 30. 3 = light / 10 = heavy.",
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=2, help="scale 2 or 4. Default is 2."
    )
    parser.add_argument(
        "-t", "--temp_dir", help="temp directory. Default is tempfile.gettempdir()."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Number of minutes to upscale per batch. Usually 1438 frames per minute. Default is 1.",
    )
    parser.add_argument(
        "-r",
        "--resume_processing",
        action="store_true",
        help="Does not purge any data in temp_dir when restarting",
    )
    parser.add_argument(
        "-x",
        "--extract_only",
        action="store_true",
        help="Exits after frame extraction. Used in conjunction with --resume_processing. You may want to run test_image.py on some extracted png files to sample what denoise level to apply if needed.",
    )
    parser.add_argument(
        "-l", "--log_level", type=int, help="logging level. logging.INFO is default"
    )
    parser.add_argument("-d", "--log_dir", help="logging directory. logging directory")
    args = parser.parse_args()

    process_file(
        args.input_file,
        args.output_file,
        args.ffmpeg,
        args.ffmpeg_encoder,
        args.scale,
        args.temp_dir,
        args.batch_size,
        args.resume_processing,
        args.extract_only,
        args.anime,
        args.denoise,
        args.log_level,
        args.log_dir,
    )
