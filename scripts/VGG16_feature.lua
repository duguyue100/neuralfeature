package.path = package.path .. ";../?.lua"

require 'loadcaffe'
require 'xlua'
require 'optim'
require 'image'
require 'hdf5';

require 'neuralfeature'

-- prepare data

prototxt='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers_deploy.prototxt.txt'
binary='/home/arlmonster/workspace/neuralfeature/data/VGG_ILSVRC_16_layers.caffemodel'

angry_images=neuralfeature.loadimagelist("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/angry.txt");
disgust_images=neuralfeature.loadimagelist("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/disgust.txt");
fear_images=neuralfeature.loadimagelist("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/fear.txt");
happy_images=neuralfeature.loadimagelist("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/happy.txt");
neutral_images=neuralfeature.loadimagelist("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/neutral.txt");
sad_images=neuralfeature.loadimagelist("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/sad.txt");
surprise_images=neuralfeature.loadimagelist("/home/arlmonster/workspace/neuralfeature/data/Train_SFEW_2_0/surprise.txt");

-- load as net

net = loadcaffe.load(prototxt, binary)
-- switch off dropout
net:evaluate();

function extract_feature(net, list)
    local out, _=neuralfeature.extract(net, list)
    
    local out_save=torch.zeros(#list, 1000);
    for i=1, #list do
        out_save[{ i, {} }]=out[i]    
    end
    
    return out_save;
end

angry_out=extract_feature(net, angry_images);
disgust_out=extract_feature(net, disgust_images);
fear_out=extract_feature(net, fear_images);
happy_out=extract_feature(net, happy_images);
neutral_out=extract_feature(net, neutral_images);
sad_out=extract_feature(net, sad_images);
surprise_out=extract_feature(net, surprise_images);

function save_file(file_adr, out_save)
    local my_file=hdf5.open(file_adr, 'w');
    my_file:write(file_adr, out_save);
    my_file:close()
end

save_file("/home/arlmonster/angry.h5", angry_out);
save_file("/home/arlmonster/disgust.h5", disgust_out);
save_file("/home/arlmonster/fear.h5", fear_out);
save_file("/home/arlmonster/happy.h5", happy_out);
save_file("/home/arlmonster/neutral.h5", neutral_out);
save_file("/home/arlmonster/sad.h5", sad_out);
save_file("/home/arlmonster/surprise.h5", surprise_out);
