--[[
Author: Yuhuang Hu
Contact: duguyue100@gmail.com

Note: This file provide a set of tools of extracting features
]]

local image=require 'image';

neuralfeature={};

--[[The function extracts features from choosen layer

Parameters
----------
net_model : object
	a neural network model
image_adds : table
	a list of path of images
layer_id : integer
	chosen layer of the network 
	
Returns
-------
out_feature : table
	list of extracted features
out_labels : table
	list of predicted labels by pre-trained model
]]
function neuralfeature.extract(net_model, image_adds, layer_id)

	-- default working with last fc layer
	local layer_id = layer_id or #net.modules-1;
	
	out_feature={};
	out_labels={};
	
	for i=1, #image_adds do
		I=image.load(image_adds[i]);
		_, y=net:forward(I:float():cuda()):max(1);
		I_out=net.modules[layer_id].output;
		table.insert(out_feature, I_out);
		table.insert(out_labels, y);
	end
		
	return out_feature, out_labels;
end