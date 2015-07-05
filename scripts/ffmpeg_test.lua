-- FFMPEG test

package.path = package.path .. ";../?.lua"

require 'loadcaffe'
require 'ffmpeg'
require 'image'

require 'neuralfeature'

prototxt='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers_deploy.prototxt.txt'
binary='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers.caffemodel'

net = loadcaffe.load(prototxt, binary)

out_feature, _=neuralfeature.v_extract(net, '../data/000046280.avi');

print(#out_feature)

print(out_feature[1])