ffmpeg -framerate 30 -i output%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4

