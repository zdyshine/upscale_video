# Upscale video 2x or 4x

Upscales video 2x or 4x using AI.

Compact Pretrained Models from the wiki below are used.

https://upscale.wiki/wiki/Model_Database#Real-ESRGAN_Compact_Custom_Models

This script will convert a video file using one minute's worth of frames at a time to save disk space. The video fragments are then merged at the end of the process into a final video file.

# You must have ffmpeg and a vulkan compatible GPU installed

Without a GPU this process would take weeks to process a 2 hour video

Make sure you are passing in a ffmpeg encoder that is compatible with your system.
The default is "av1_qsv" since I'm using an Intel Arc A750 GPU which supports AV1 hardware encoding.

https://ffmpeg.org/ffmpeg-codecs.html#Video-Encoders.

# Installation

Download - https://ffmpeg.org/download.html

pip install ncnn_vulkan

pip install numpy

pip install wakepy

git clone https://github.com/davlee1972/upscale_video.git

# Usage

--resume_processing can be used after you abort a run and want to pickup where you left off.

--extract_only will stop processing after frames extraction. You may want to run test_image.py on some
images to test noise removal and then restart processing with --resume_processing with --denoise level.

```console
Usage: python upscale_video.py -i infile -f ffmpeg_location

  -h                   Show this help
  -i input_file        Input video file
  -o output_file       Optional output video file location. Default is input_file + (".2x." or ".4x.")
  -f ffmpeg            Location of ffmpeg
  -e ffmpeg_encoder    ffmpeg encoder. Default is av1_qsv.
                       Full list is here https://ffmpeg.org/ffmpeg-codecs.html#Video-Encoders.
  -a anime             Adds additional processing for anime videos to remove grain and color bleeding.
  -n denoise           Adds additional processing to remove film grain.
                       Denoise level 1 to 30. 3 = light / 10 = heavy.
  -s scale             2 or 4. Default is 2.
  -t temp_dir          temp directory where upscale_video folder is created.
                       Default is tempfile.gettempdir().
                       For windows this would be the %temp% environment variable.
  -b batch_size        Number of minutes to upscale per batch. Usually 1438 frames / min. Default is 1.
  -r resume_processing Does not purge any data in temp_dir/upscale_video when restarting.
  -x extract_only      Exits after frame extraction.
                       Used in conjunction with --resume_processing. You may want to run test_image.py on
                       some extracted png files to sample what denoise level to apply if needed.
  -l log_level         Logging level. logging.INFO is default.
  -d log_dir           Optional directory to write final log file.

```

```console
Usage: python test_image.py -i infile

  -h                   Show this help
  -i input_file        Input image file
  -o output_file       Optional output image file location. Default is input_file + (".2x." or ".4x.")
  -a anime             Adds additional processing for anime images to remove grain and color bleeding.
  -n denoise           Adds additional processing to remove image grain.
                       Denoise level 1 to 30. 3 = light / 10 = heavy.
  -s scale             2 or 4. Default is 2.
```

# Samples

![alt text](https://i.imgur.com/nkbA0Ft.png)
Original 1920 x 800 extracted image from Underworld Blu-ray

![alt text](https://i.imgur.com/Z2djqQN.png)
Upscaled 2x using --scale 2. Took 40 hours to process 200,000+ frames.

![alt text](https://i.imgur.com/GOFMK47.png)
Upscaled 2x with light denoise using --scale 2 --denoise 3. Denoise added an additional 3 hours of processing.

# Notes

This python code is used to scale my existing 2k bluray collecion to 4k

Source framerates are assumed to be more or less consistent. 
A frames per second calculation is performed taking total number of frames / duration.
It should come out to 23.976 fps for most movies.
This fps is used to reassemble the upscaled png images into a video file at the very end.
