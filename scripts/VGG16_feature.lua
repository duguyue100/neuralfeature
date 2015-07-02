require 'loadcaffe'
require 'xlua'
require 'optim'
require 'image'

require 'neuralfeature'

-- prepare data

prototxt='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers_deploy.prototxt.txt'
binary='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers.caffemodel'

-- load as net

I=image.load("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/Angry/Airheads_000519240_00000005.png");


net = loadcaffe.load(prototxt, binary)

--_, y=net:forward(I:float():cuda());

--print(net.modules[#net.modules-1].output)

out=neuralfeature.extract(net, {"/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/Angry/Airheads_000519240_00000005.png"})

print(out[#out])