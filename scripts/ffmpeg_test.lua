-- FFMPEG test

require 'ffmpeg'

vid=ffmpeg.Video('../data/000046280.avi');

vid:play{}

