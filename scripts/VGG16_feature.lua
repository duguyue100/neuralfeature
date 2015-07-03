package.path = package.path .. ";../?.lua"

require 'loadcaffe'
require 'xlua'
require 'optim'
require 'image'

require 'neuralfeature'

-- prepare data

prototxt='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers_deploy.prototxt.txt'
binary='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers.caffemodel'

-- load as net

file=io.open("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/Angry/angry.txt", 'r');
io.input(file);

angry_images={};

while true do
	local image_adr=io.read();	
	if image_adr==nil then break end	
	table.insert(angry_images, image_adr);
end

net = loadcaffe.load(prototxt, binary)

out, labels=neuralfeature.extract(net, angry_images)

print(out[#out])
