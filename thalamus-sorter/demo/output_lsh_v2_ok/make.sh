ffmpeg -framerate 30 -i input_output%06d.png -c:v libx264 -pix_fmt yuv420p input_output.mp4
ffmpeg -framerate 30 -i weights_output%06d.png -c:v libx264 -pix_fmt yuv420p weights_output.mp4

