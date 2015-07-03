package.path = package.path .. ";../?.lua"

require 'loadcaffe'
require 'xlua'
require 'optim'
require 'image'

require 'neuralfeature'

-- prepare data

prototxt='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers_deploy.prototxt.txt'
binary='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers.caffemodel'

angry_images=neuralfeature.loadimagelist("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/Angry/angry.txt");

-- load as net

net = loadcaffe.load(prototxt, binary)
-- switch off dropout
net:evaluate();

out, labels=neuralfeature.extract(net, angry_images)

print(out[#out])
