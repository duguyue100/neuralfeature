# Neural Feature
Extract feature from pre-trained deep model using Torch 7

## Updates

+ update function

## Notes

### Reading Videos in Torch 7

You can use `ffmpeg` package for reading videos in Torch 7, basically, you can first install the library by:

```
luarocks install ffmpeg
```

After the installation, you may find that you cannot read video because of lacking `ffmpeg` in your system (Torch 7's package and `ffmepg` build are different), Ubuntu 14.04 cannot find `ffmepg` because it removed in their official package maintain, you can install `ffmpeg` by following commands though:

```
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get install ffmpeg
```

Now you can try this simple script to test your installation

```lua
require 'ffmpeg'

vid=ffmpeg.Video('/path/to/your/video');

vid:play{}
```

### VGG ILSVRC-2014 16-layer

```
conv1_1: 64 3 3 3
conv1_2: 64 64 3 3
conv2_1: 128 64 3 3
conv2_2: 128 128 3 3
conv3_1: 256 128 3 3
conv3_2: 256 256 3 3
conv3_3: 256 256 3 3
conv4_1: 512 256 3 3
conv4_2: 512 512 3 3
conv4_3: 512 512 3 3
conv5_1: 512 512 3 3
conv5_2: 512 512 3 3
conv5_3: 512 512 3 3
fc6: 1 1 25088 4096
fc7: 1 1 4096 4096
fc8: 1 1 4096 1000
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> (24) -> (25) -> (26) -> (27) -> (28) -> (29) -> (30) -> (31) -> (32) -> (33) -> (34) -> (35) -> (36) -> (37) -> (38) -> (39) -> (40) -> output]
  (1): nn.SpatialConvolutionMM
  (2): nn.ReLU
  (3): nn.SpatialConvolutionMM
  (4): nn.ReLU
  (5): inn.SpatialMaxPooling
  (6): nn.SpatialConvolutionMM
  (7): nn.ReLU
  (8): nn.SpatialConvolutionMM
  (9): nn.ReLU
  (10): inn.SpatialMaxPooling
  (11): nn.SpatialConvolutionMM
  (12): nn.ReLU
  (13): nn.SpatialConvolutionMM
  (14): nn.ReLU
  (15): nn.SpatialConvolutionMM
  (16): nn.ReLU
  (17): inn.SpatialMaxPooling
  (18): nn.SpatialConvolutionMM
  (19): nn.ReLU
  (20): nn.SpatialConvolutionMM
  (21): nn.ReLU
  (22): nn.SpatialConvolutionMM
  (23): nn.ReLU
  (24): inn.SpatialMaxPooling
  (25): nn.SpatialConvolutionMM
  (26): nn.ReLU
  (27): nn.SpatialConvolutionMM
  (28): nn.ReLU
  (29): nn.SpatialConvolutionMM
  (30): nn.ReLU
  (31): inn.SpatialMaxPooling
  (32): nn.View
  (33): nn.Linear(25088 -> 4096)
  (34): nn.ReLU
  (35): nn.Dropout
  (36): nn.Linear(4096 -> 4096)
  (37): nn.ReLU
  (38): nn.Dropout
  (39): nn.Linear(4096 -> 1000)
  (40): nn.SoftMax
}
```

## Contacts

Yuhuang Hu  
Advanced Robotic Lab  
Department of Artificial Intelligence  
Faculty of Computer Science & IT  
University of Malaya  
Email: duguyue100@gmail.com
