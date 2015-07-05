-- FFMPEG test

require 'ffmpeg'
require 'image'

vid=ffmpeg.Video('../data/000046280.avi');

-- print number of frames

print(vid.nframes);

-- dumpy all as tensor

content=vid:totensor{}

-- display tensor
image.display(content[10])

-- play the video

vid:play{}

-- for more information, please read code of:
-- https://github.com/clementfarabet/lua---ffmpeg/blob/master/init.lua
