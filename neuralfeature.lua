--[[
Author: Yuhuang Hu
Contact: duguyue100@gmail.com

Note: This file provide a set of tools of extracting features
]]

local image=require 'image';

neuralfeature={};

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