--[[
Author: Yuhuang Hu
Contact: duguyue100@gmail.com

Note: This file provide a set of tools of extracting features
]]

local image=require 'image';
local ffmpeg=require 'ffmpeg';

neuralfeature={};

---The function extracts features from choosen layer
--
-- Parameters
-- ----------
-- net_model : object
--     a neural network model
-- image_adds : table
--     a list of path of images
-- layer_id : integer
--     chosen layer of the network 
--
-- Returns
-- -------
-- out_feature : table, elements are floatTensor
--     list of extracted features
-- out_labels : table
--	   list of predicted labels by pre-trained model
--
function neuralfeature.extract(net_model, image_adds, layer_id)

    -- default working with last fc layer
    local layer_id = layer_id or #net.modules-1;
    
    out_feature={};
    out_labels={};
    
    for i=1, #image_adds do
        I=image.load(image_adds[i]);
        _, y=net:forward(I:float():cuda()):max(1);
        I_out=net.modules[layer_id].output;
        
        --print(I_out:float()[{ {1,10} }])
        table.insert(out_feature, I_out:float());
        table.insert(out_labels, y);
    end
    	
    return out_feature, out_labels;
end

--- extract video features using ffmpeg
-- 
-- Parameters
-- ----------
-- net_model : object
--     a neural network model
-- video_add : string
--     path of a video
-- layer_id : integer
--     chosen layer of the network 
-- 
-- Returns
-- -------
-- out_feature : table, elements are floatTensor
--     list of extracted features
-- out_labels : table
--     list of predicted labels by pre-trained model
--
function neuralfeature.v_extract(net_model, video_add, layer_id)
    -- default working with last fc layer
    local layer_id = layer_id or #net.modules-1;
    
    out_feature={};
    out_labels={};
    
    vid=ffmpeg.Video(video_add);
    frames=vid:totensor{}
    
    for i=1, vid.nframes do
        local I=image.scale(frames[i], 224, 224);
        
        _, y=net:forward(I:float():cuda()):max(1);
        I_out=net.modules[layer_id].output;
        table.insert(out_feature, I_out:float());
        table.insert(out_labels, y);
    end
    
    return out_feature, out_labels;
end


--- Load a list of image paths given a text file
--
-- Parameters
-- ----------
-- list_adr : string
--     file path of the image list 
--
-- Returns
-- -------
-- image_list : table
--     list of image paths
-- 
function neuralfeature.loadimagelist(list_adr)
    f=io.open(list_adr, 'r');
    io.input(f);
    image_list={};
    while true do
        local image_adr=io.read();  
        if image_adr==nil then break end    
        table.insert(image_list, image_adr);
    end
    
    return image_list;
end