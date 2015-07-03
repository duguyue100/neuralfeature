
require 'hdf5';

data=torch.rand(5,5);

print(data)

local myFile=hdf5.open('/home/arlmonster/test.h5', 'w');
myFile:write('/home/arlmonster/test.h5', data);

myFile:close()

local myFile = hdf5.open('/home/arlmonster/test.h5', 'r');
datax = myFile:read('/home/arlmonster/test.h5'):all()

print(datax)
myFile:close()