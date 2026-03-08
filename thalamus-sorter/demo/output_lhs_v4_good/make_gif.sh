#ffmpeg -f image2 -framerate 24 -i output%06d.png output.gif
ffmpeg -f image2 -framerate 24 -i output%06d.png -vf "scale=-1:384" output_small.gif


