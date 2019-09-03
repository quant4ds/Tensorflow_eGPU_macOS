# Tensorflow 1.14 on macOS with Nvidia GPU Support
(by quant4ds; 9/1/2019)
<!-- OSX10.13.6-17G2208_Xcode9.4.1_tensorflow1.14_python3.7.3_CUDA10.0_cuDNN7.4.2 -->

## Table of Contents
- [Motivation][101]
- [Environment][102]
- [Hardware][103]
- [BACKUP][104]
- [macOS][105]
- [Nvidia eGPU Installation][106]
    + [eGPU][107]
    + [Xcode][108]
    + [CUDA][109]
    + [cuDNN][110]
    + [(optional) NCCL][111]
- [eGPU Test][112]
- [Bootcamp Windows][113]
- [Tensorflow-GPU][114]
    + [Self-compilation][117]
- [Tensorflow Test][118]

## Motivation
The unfortunate battle between Apple and Nvidia sets obstacles for Mac users to utilize majority of Nvidia GPU with their machines. What exacerbates our lives is the announcement from tensorflow team to stop support macOS after v1.2. 

It's been more than a year since the announcement of Mojave in 2018. The web driver for their GPUs under the new macOS has not been published by Nvidia. And it's unknown when, if at all, such driver would be available for 10.14 or 10.15. 

Meanwhile, however, effortless work has been done in the unofficial way by multiple communities and individuals. Their work has succefully bridged the gaps to some level. After extensive amount of research and experiment, I am able aggregate necessary information to benefit from their generous contributions. 

This guide is to take notes of my experience in the complete process to connect a Nvidia GeForce GTX card as an eGPU to work with my MBP, and enable tensorflow to leverage the eGPU in machine learning. At the same time, it supplements other guidance from the internet by incorporating the latest available tools (i.e., tensorflow 1.14, CUDA 10, cuDNN 7.4, MBP 2019) and pave a way of "can-do". 

## Environment
- MacBook Pro (13-inch, 2019, Four Thunderbolt 3 ports) 
- Asus Strix Gaming X GTX 1080 Ti
- Razer Core X Chroma
- macOS High Sierra 10.13.6 build 17G8030
- Xcode 9.4.1
- CUDA Toolkit 10.0
- cuDNN 7.4.2
- Python 3.7.3 (coming from Anaconda3-2019.07)
- Tensorflow 1.14

## Hardware
- This guide should be applicable to both 2018 and 2019 MBPs with touch bar
- Other models of macs are possible to work with more or less change in the process. For details, refer to [egpu.io](https://egpu.io) community. You may want to pay special attention if your mac has 
    + TB1/2 ports instead of TB3; or
    + dGPU in addition to iGPU.
- For the choice of the enclosure, I personally locked at Mantiz Venus and Razer Core X Chroma mainly for taking them as USB hub and the 1-cable solution. You may have different preference for your situation; it's highly YMMV. Check the [enclosure guide](https://egpu.io/best-egpu-buyers-guide/#tb3-enclosures) for details. 
- I intentionally avoided 20 series cards, given some discussions about the RTX [issues](https://www.nvidia.com/en-us/geforce/forums/geforce-graphics-cards/5/296479/another-dead-rtx-2080ti-with-an-explanation/). 

## BACKUP
```diff
- Back up!!
```

## macOS
If your current macOS is High Sierra 10.13.6, you can skip this section and proceed directly to [eGPU installation](https://github.com/quant4ds/tensorflow_gpu_macOS#nvidia-egpu-installation). 

The mid-2019 MBP comes with Mojave (macOS 10.14.6) pre-installed. Despite the nice dark mode offered, due to the lack of compatible Nvidia web driver, I decided that I had to give it up and embrace High Sierra (macOS 10.13.6). At this moment, the latest available Nvidia web driver is for build 17G8030 (all available versions can be found [here](https://www.insanelymac.com/forum/topic/324195-nvidia-web-driver-updates-for-macos-high-sierra-update-july-30-2019/), [here](https://www.tonymacx86.com/nvidia-drivers/), and [here](http://www.macvidcards.com/drivers.html)). 

<details>
  <summary>Download macOS 10.13.6</summary>
  
  - **For 2018 and 2019 MBPs with touch bar, you have to download a special built of the High Sierra (i.e., 17G2208).** 
      + It's reported that regular build High Sierra (i.e., 17G65) is not compatible with 2018 MBP with touch bar. I tested on my 2019 MBP and it's also no-go. 
  - Four ways to download: 
      + Direct download link if you can find one - I used this [17G2208](http://oceanofdmg.com/download-macos-high-sierra-v10-13-6-17g2208-app-store-dmg/); 
      + Use the [patcher](http://dosdude1.com/highsierra/) (only for regular build 17G65);
      + First directly download the compact downloader from [app store](https://apps.apple.com/us/app/macos-high-sierra/id1246284741?mt=12), followed by modifying the file as shown in [这里](https://www.newlearner.site/2019/07/22/full-size-macos.html) (only for regular build 17G65); and
      + Download [installinstallmacos.py](https://github.com/munki/macadmin-scripts/blob/master/installinstallmacos.py) and follow the [instruction](https://github.com/munki/macadmin-scripts/blob/master/docs/installinstallmacos.md) (for all latest versions including both 17G2208 and 17G65). 
  - Verify the installer authenticity
      + Double click the to mount the .dmg file
      + Run below command in terminal to get Sha1 checksum of the installer
      ```bash
      $ shasum /Volumes/macOS.High.Sierra.10.13.6/Install*OS*.app/Contents/SharedSupport/{Base,Install}*.dmg
      ```
      + Compare against [here](https://github.com/notpeter/apple-installer-checksums)
</details>

<details>
  <summary>Create bootable flash drive</summary>
  
  - **The flash drive should be at least 8gb in capacity.**
  - Use Disk Utilities to erase the flash drive as "Mac OS Extended (Journaled)" format and named as "HighSierra" (or any name you wish); 
  - Double click to mount the High Sierra .dmg file that you have just downloaded or created; 
  - Run below commands in terminal:
  ```bash
  $ cd /Volumes/macOS.High.Sierra.10.13.6 
  $ sudo /Volumes/macOS.High.Sierra.10.13.6/Install\ macOS\ High\ Sierra.app/Contents/Resources/createinstallmedia --volume /Volumes/HighSierra --applicationpath /Volumes/macOS.High.Sierra.10.13.6/Install\ macOS\ High\ Sierra.app --nointeraction
  ```
  - The flash drive name changes to “Install macOS High Sierra”.
</details>

<details>
  <summary>Install High Sierra</summary>
  
  - First make sure the mac can be booted from external sources:
      + Restart the mac while pressing Command+R and holding until entering recovery mode;
      + Turn on "Allow Booting From External Media" under the Startup Security Utility in menu. 
  - Plug in the bootable flash drive and restart the mac, while pressing the Option key and holding until start option shows;
  - Select "Install macOS High Sierra" and enter; 
  - In the recovery mode, erase the internal SSD as "APFS" format and name it "Macintosh HD" as convention (or any other name you wish);
      + When erasing, select the **whole SSD ("APPLE HDD...")**;
      + Do not prefer to select "Case Sensitive" or "Encrypted" format for convenience.
  - Install... 
      + When the fresh installation completes, the build code shows as 17G2208;
      + Run updates from app store;
      + now the build code shows as 17G8030.
</details>

## Nvidia eGPU Installation
> ### eGPU

<details>
  <summary>Turn off SIP</summary>
 
  - Restart the mac while pressing Command+R and holding until entering recovery mode;
  - Select "Terminal" under Utilities in menu, and run below command:
  ```bash
  $ csrutil disable
  ```
</details>

<details>
  <summary>purge-wrangler</summary>
  
  - Restart the mac while the eGPU is plugged in;
  - Run below commands in terminal (will be asked to restart after running):
  ```bash
  $ curl -s "https://api.github.com/repos/mayankk2308/purge-wrangler/releases/latest" | grep '"browser_download_url":' | sed -E 's/.*"([^"]+)".*/\1/' | xargs curl -L -s -0 > purge-wrangler.sh && chmod +x purge-wrangler.sh && ./purge-wrangler.sh && rm purge-wrangler.sh
  ```
      + It automatically installs Nvidia web driver (currently v387.10.10.10.40.130), which won’t succeed when trying to install manually on macOS 10.13.6 build 17G2208.
  - Source code, instuction, and discussions can be found [here](https://github.com/mayankk2308/purge-wrangler), and [here](https://egpu.io/purge-wrangler.sh).

  ---
  **Note**:  
  An alternative to purge-wrangler is [macOS-eGPU](https://github.com/learex/macOS-eGPU). Instruction can be found [here](https://theunlockr.com/how-to-use-nvidia-cards-with-your-mac-egpu/). However the former is recommended. 
  ---
</details>

<details>
  <summary>System info</summary>

  - About this mac:  
  ![](/misc/system_overview.png)  
  ![](/misc/system_displays.png)  
  - System report:  
  ![](/misc/system_tb3.png)  
  ![](/misc/system_graphics.png)  
</details>

> ### Xcode

<details>
  <summary>Download and install</summary>
 
  - Download Xcode 9.4.1 from [apple developer site](https://developer.apple.com/download/more/). 
      + Xcode 10.0 seems to be the latest version that can be installed on High Sierra without and mod. However, **it will cause error in CUDA installation** (tested; something similar is also mentioned [这里](https://segmentfault.com/a/1190000015807229)). 
  - Run below command in terminal:
  ```bash
  $ sudo xcode-select -s /Applications/Xcode.app
  ```
</details>

<details>
  <summary>Command line tools</summary>
 
  - Download Command Line Tools for Xcode and install from [apple developer site](https://developer.apple.com/download/more/).
      + Select the version **matching macOS 10.13 and Xcode 9.4.1**
</details>

<details>
  <summary>Verify installation</summary>

  - Run below command in terminal:
  ```bash
  $ cc -v
  Apple LLVM version 9.1.0 (clang-902.0.39.2)
  Target: x86_64-apple-darwin17.7.0
  Thread model: posix
  InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
  ```
</details>

> ### CUDA

<details>
  <summary>Download and install</summary>

  - If already has CUDA installed, uninstall first by running below commands in terminal:
  ```bash
  $ sudo /usr/local/bin/uninstall_cuda_drv.pl 
  $ sudo /usr/local/cuda/bin/uninstall_cuda_10.0.pl  # change CUDA version as needed
  $ sudo rm -rf /Developer/NVIDIA/CUDA-10.0/  # change CUDA version as needed
  $ sudo rm -rf /Library/Frameworks/CUDA.framework 
  $ sudo rm -rf /usr/local/cuda/
  ```
  - CUDA driver:
      + Download version 410.130 from [here](https://www.nvidia.com/en-us/drivers/cuda/macosx-cuda-410-130-driver/) (to **match CUDA toolkit 10.0**);
      + Install...
  - CUDA toolkit:
      + Download version 10.0.130 from [here](https://developer.nvidia.com/cuda-toolkit-archive) (to **match Nvidia web driver** [installed](https://github.com/quant4ds/tensorflow_gpu_macOS#nvidia-egpu-installation) as shown [here](https://www.nvidia.com/download/driverResults.aspx/149652/));
      ```bash
      Version:  387.10.10.10.40.130
      Release Date:   2019.7.30
      Operating System:   macOS High Sierra 10.13.6
      CUDA Toolkit:   10.1
      ```
      + Install... 
</details>

<details>
  <summary>Update bash_profile</summary>

  - Run below commands in terminal to open the file:
  ```bash
  $ touch ~/.bash_profile; open ~/.bash_profile
  ```
  - Add below lines and save:
  ```bash
  export CUDA_HOME=/usr/local/cuda  
  export DYLD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/nvvm/lib:$CUDA_HOME/extras/CUPTI/lib:/usr/local/nccl/lib"  
  export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH  
  export PATH=$CUDA_HOME/bin:$PATH
  ```
      + **Special attention needs to be paid to line changer "\r\n"**
  - Run below command in terminal to apply bash_profile:
  ```bash
  $ . ~/.bash_profile
  ```

  ---
  **Note**:  
  If not using bash, then should change other file accordingly (e.g., ~/.zshrc if using zsh).
  ---
</details>

<details>
  <summary>Verify installation</summary>

  - Run below commands in terminal:
  ```bash
  $ nvcc -V
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2018 NVIDIA Corporation
  Built on Sat_Aug_25_21:08:56_CDT_2018
  Cuda compilation tools, release 10.0, V10.0.130
  ```
  and
  ```bash
  $ kextstat | grep -i cuda.
  171    0 0xffffff7f84f15000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) E13478CB-B251-3C0A-86E9-A6B56F528FE8 <4 1>
  ```
  - Sample test by running below commands in terminal:
  ```bash
  $ cd /usr/local/cuda/samples
  $ sudo make -C 1_Utilities/deviceQuery
  $ ./bin/x86_64/darwin/release/deviceQuery
  ./bin/x86_64/darwin/release/deviceQuery Starting...

   CUDA Device Query (Runtime API) version (CUDART static linking)

  Detected 1 CUDA Capable device(s)

  Device 0: "GeForce GTX 1080 Ti"
    CUDA Driver Version / Runtime Version          10.0 / 10.0
    CUDA Capability Major/Minor version number:    6.1
    Total amount of global memory:                 11264 MBytes (11810963456 bytes)
    (28) Multiprocessors, (128) CUDA Cores/MP:     3584 CUDA Cores
    GPU Max Clock rate:                            1607 MHz (1.61 GHz)
    Memory Clock rate:                             5505 Mhz
    Memory Bus Width:                              352-bit
    L2 Cache Size:                                 2883584 bytes
    Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
    Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
    Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
    Total amount of constant memory:               65536 bytes
    Total amount of shared memory per block:       49152 bytes
    Total number of registers available per block: 65536
    Warp size:                                     32
    Maximum number of threads per multiprocessor:  2048
    Maximum number of threads per block:           1024
    Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
    Maximum memory pitch:                          2147483647 bytes
    Texture alignment:                             512 bytes
    Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
    Run time limit on kernels:                     Yes
    Integrated GPU sharing Host Memory:            No
    Support host page-locked memory mapping:       Yes
    Alignment requirement for Surfaces:            Yes
    Device has ECC support:                        Disabled
    Device supports Unified Addressing (UVA):      Yes
    Device supports Compute Preemption:            Yes
    Supports Cooperative Kernel Launch:            Yes
    Supports MultiDevice Co-op Kernel Launch:      Yes
    Device PCI Domain ID / Bus ID / location ID:   0 / 67 / 0
    Compute Mode:
       < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

  deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.0, CUDA Runtime Version = 10.0, NumDevs = 1
  Result = PASS
  ```
    + Look for the "**Resule = PASS**" in the end of the output.
</details>

> ### cuDNN

<details>
  <summary>Download and install</summary>

  - Download version 7.4.2 from [here](https://developer.nvidia.com/rdp/cudnn-archive) (to **match CUDA toolkit 10.0**);
  - Run below command in terminal:
      + Copy:
      ```bash
      $ sudo cp ~/Downloads/cudnn-10.0-osx-x64-v7.4.2.24.tar /Developer/
      ```
      + Uncompress:
      ```bash
      $ sudo tar -xvf /Developer/cudnn-10.0-osx-x64-v7.4.2.24.tar -C /Developer/
      ```
      + Combine cuDNN with CUDA:
      ```bash
      $ sudo mv /Developer/cuda/include/cudnn.h /Developer/NVIDIA/CUDA-10.0/include/ 
      $ sudo mv /Developer/cuda/lib/libcudnn* /Developer/NVIDIA/CUDA-10.0/lib/ 
      $ sudo ln -s /Developer/NVIDIA/CUDA-10.0/include/* /usr/local/cuda/include/ 
      $ sudo ln -s /Developer/NVIDIA/CUDA-10.0/lib/* /usr/local/cuda/lib/
      ```
</details>

> ### (optional) NCCL

<details>
  <summary>Download and install</summary>

  - Download version 2.4.8 from [here](https://developer.nvidia.com/nccl/nccl-download) (to **match CUDA toolkit 10.0**);
      + Need "The Unarchiver" from app store to uncompress the .txz archive
  - Run below commands in terminal:
  ```bash
  $ sudo mkdir -p /usr/local/nccl 
  $ sudo cp -a /Volumes/nccl_2.4.8-1+cuda10.0_x86_64/* /usr/local/nccl 
  $ sudo mkdir -p /usr/local/include/third_party/nccl  
  $ sudo ln -s /usr/local/nccl/include/nccl.h /usr/local/include/third_party/nccl  
  ```

  ---
  **Note**:  
  Probably only needed for MBP15 with dGPU.
  ---
</details>

## eGPU Test

<details>
  <summary>CUDA-Z</summary>

  - Install
      + Install Homebrew by running below command in terminal:
      ```bash
      $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
      ```
      + Brew install CUDA-Z by running below command in terminal:
      ```bash
      $ brew cask install cuda-z
      ```
  - Result  
  ![](/misc/guda_z.png)

</details>

<details>
  <summary>Unigine</summary>

  - [Valley](https://assets.unigine.com/d/Unigine_Valley-1.0.dmg)  
  ![](/misc/unigine_valley.png)
  - [Heaven](https://assets.unigine.com/d/Unigine_Heaven-4.0.dmg)  
  ![](/misc/unigine_heaven.png)
</details>

<details>
  <summary>Geekbench</summary>

  ![](/misc/geekbench4_opencl.png)  
  ![](/misc/geekbench4_metal.png)
</details>

## Bootcamp Windows

<details>
  <summary>TODO</summary>

  - https://egpu.io/boot-camp-egpu-setup-guide/
  - https://egpu.io/forums/mac-setup/automate-egpu-efi-egpu-boot-manager-for-macos-and-windows/
  - https://blog.csdn.net/ssujoensiang/article/details/78620616
</details>

## Tensorflow-GPU

<details>
  <summary>Anaconda/Python</summary>

  - Download and install Anaconda3 2019.07 from [here](https://repo.anaconda.com/archive/Anaconda3-2019.07-MacOSX-x86_64.pkg);
      + Python 3.7.3 is included
  - Run below command in terminal to create a virtual environment:
  ```bash
  $ conda create —name tf_gpu python=3.7
  ```
</details>

<details>
  <summary>Pre-compiled tensorflow wheel</summary>

  - Download "tensorflow-1.14.0rc1-py27-py37-cuda10-cudnn74-full" from [here](https://github.com/TomHeaven/tensorflow-osx-build/releases/tag/v1.14.0rc1_cu100);
      + Compilation elements **match all environment/driver/software versions**.
  - Run below commands in terminal to install tensorflow:
  ```bash
  $ conda activate tf_gpu 
  $ pip install ~/Downloads/tensorflow-1.14.0rc1-cp37-cp37m-macosx_10_13_x86_64.whl
  ```
</details>

> ### Self-compilation

<details>
  <summary>TODO</summary>

  - homebrew
  - llvm
  - bazel
  - https://github.com/TomHeaven/tensorflow-osx-build/blob/master/build_instructions_1.10.md
  - https://medium.com/xplore-ai/nvidia-egpu-macos-tensorflow-gpu-the-definitive-setup-guide-to-avoid-headaches-f40e831f26ea
</details>

## Tensorflow Test

<details>
  <summary>Test 1 (device detection)</summary>

  Run below python code:
  ```python
  from tensorflow.python.client import device_lib
  print(device_lib.list_local_devices())
  ```
  We get:
  ```python
  2019-09-03 12:50:43.381079: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.10.0.dylib
  2019-09-03 12:50:45.565103: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcuda.dylib
  2019-09-03 12:50:45.570169: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:50:45.570297: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1640] Found device 0 with properties: 
  name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.607
  pciBusID: 0000:43:00.0
  2019-09-03 12:50:45.570393: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudart.10.0.dylib
  2019-09-03 12:50:45.613619: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcublas.10.0.dylib
  2019-09-03 12:50:45.647913: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcufft.10.0.dylib
  2019-09-03 12:50:45.663483: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcurand.10.0.dylib
  2019-09-03 12:50:45.720211: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcusolver.10.0.  dylib
  2019-09-03 12:50:45.755571: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcusparse.10.0.  dylib
  2019-09-03 12:50:45.798164: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudnn.7.dylib
  2019-09-03 12:50:45.798272: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:50:45.798491: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:50:45.798595: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1763] Adding visible gpu devices: 0
  2019-09-03 12:50:45.798732: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudart.10.0.dylib
  2019-09-03 12:50:47.160671: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1181] Device interconnect StreamExecutor with strength 1 edge matrix:
  2019-09-03 12:50:47.160684: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1187]      0 
  2019-09-03 12:50:47.160687: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1200] 0:   N 
  2019-09-03 12:50:47.160858: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:50:47.161059: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:50:47.161228: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:50:47.161349: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1326] Created TensorFlow device (/device:GPU:0 with 8264 MB memory) ->   physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:43:00.  0, compute capability: 6.1)
  [name: "/device:CPU:0"
  device_type: "CPU"
  memory_limit: 268435456
  locality {
  }
  incarnation: 14163670328676646715
  , name: "/device:GPU:0"
  device_type: "GPU"
  memory_limit: 8665955328
  locality {
    bus_id: 1
    links {
    }
  }
  incarnation: 11986345720044061
  physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id:   0000:43:00.0, compute capability: 6.1"
  ]
  
  Process finished with exit code 0
  ```
</details>

<details>
  <summary>Test 2 (a simple calculation)</summary>

  Run below python code:
  ```python
  import tensorflow as tf
  
  config = tf.ConfigProto()
  config.log_device_placement = True
  
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)

  with tf.Session(config=config) as sess:
      print(sess.run(c))
  ```
  We get:
  ```python
  2019-09-03 12:56:57.447240: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.10.0.dylib
  WARNING:tensorflow:From /Users/alex/Documents/workon/tf_gpu_test/gpu_test1.py:3  : The name tf.ConfigProto is deprecated. Please use tf.compat.v1.  ConfigProto instead.
  
  WARNING:tensorflow:From /Users/alex/Documents/workon/tf_gpu_test/gpu_test1.py:  11: The name tf.Session is deprecated. Please use tf.compat.v1.Session   instead.
  
  2019-09-03 12:56:58.667338: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcuda.dylib
  2019-09-03 12:56:58.672698: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:56:58.672828: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1640] Found device 0 with properties: 
  name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.607
  pciBusID: 0000:43:00.0
  2019-09-03 12:56:58.672924: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudart.10.0.dylib
  2019-09-03 12:56:58.682121: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcublas.10.0.dylib
  2019-09-03 12:56:58.688366: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcufft.10.0.dylib
  2019-09-03 12:56:58.690745: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcurand.10.0.dylib
  2019-09-03 12:56:58.701561: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcusolver.10.0.  dylib
  2019-09-03 12:56:58.711459: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcusparse.10.0.  dylib
  2019-09-03 12:56:58.723670: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudnn.7.dylib
  2019-09-03 12:56:58.723813: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:56:58.724151: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:56:58.724308: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1763] Adding visible gpu devices: 0
  2019-09-03 12:56:58.724472: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudart.10.0.dylib
  2019-09-03 12:56:59.301986: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1181] Device interconnect StreamExecutor with strength 1 edge matrix:
  2019-09-03 12:56:59.301999: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1187]      0 
  2019-09-03 12:56:59.302002: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1200] 0:   N 
  2019-09-03 12:56:59.302150: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:56:59.302348: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:56:59.302517: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:56:59.302639: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU  :0 with 8264 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080   Ti, pci bus id: 0000:43:00.0, compute capability: 6.1)
  2019-09-03 12:56:59.303165: I tensorflow/core/common_runtime/direct_session.cc:  296] Device mapping:
  /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX   1080 Ti, pci bus id: 0000:43:00.0, compute capability: 6.1
  
  2019-09-03 12:56:59.303711: I tensorflow/core/common_runtime/placer.cc:54]   MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
  2019-09-03 12:56:59.303719: I tensorflow/core/common_runtime/placer.cc:54] a: (  Const)/job:localhost/replica:0/task:0/device:GPU:0
  2019-09-03 12:56:59.303725: I tensorflow/core/common_runtime/placer.cc:54] b: (  Const)/job:localhost/replica:0/task:0/device:GPU:0
  Device mapping:
  /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX   1080 Ti, pci bus id: 0000:43:00.0, compute capability: 6.1
  MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
  a: (Const): /job:localhost/replica:0/task:0/device:GPU:0
  b: (Const): /job:localhost/replica:0/task:0/device:GPU:0
  [[22. 28.]
   [49. 64.]]
  
  Process finished with exit code 0
  ```


</details>

<details>
  <summary>Test 3 (a model training)</summary>

  Run below python code:
  ```python
  import tensorflow as tf
  import tensorflow.keras as keras
  import time
  
  class myCallback(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
          if(logs.get("acc")>0.998):
              print("\nStop as accuracy on training data is over 99.8%")
              self.model.stop_training = True

  mnist = keras.datasets.mnist
  (image_train, label_train), (image_test, label_test) = mnist.load_data()
  image_train, image_test = image_train[: ,: ,: ,None]/255, image_test[: ,: ,: ,None]/255
  
  with tf.device("/gpu:0"):
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True 
      config.gpu_options.per_process_gpu_memory_fraction = 0.8 
      sess = tf.Session(config=config)
      keras.backend.set_session(sess)
  
      model = tf.keras.models.Sequential([
          keras.layers.Conv2D(64, [3,3], activation="relu", input_shape=[28,28,1]  ),
          keras.layers.MaxPooling2D([2,2]),
          keras.layers.Flatten(),
          keras.layers.Dense(128, activation="relu"),
          keras.layers.Dense(10, activation="softmax")
      ])
  
      model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["acc"])
  
      start_time = time.time()
      model.fit(image_train, label_train, epochs=20, callbacks=[myCallback()])
      duration = time.time() - start_time
      print('Training Duration %.3f sec' % (duration))
  
  model.evaluate(image_test, label_test)

  keras.backend.clear_session()
  ```
  We get:
  ```python
  2019-09-03 12:59:54.396877: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.10.0.dylib
  WARNING:tensorflow:From /Users/alex/Documents/workon/tf_gpu_test/gpu_test3.py:  30: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.  ConfigProto instead.
  
  WARNING:tensorflow:From /Users/alex/Documents/workon/tf_gpu_test/gpu_test3.py:  33: The name tf.Session is deprecated. Please use tf.compat.v1.Session   instead.
  
  2019-09-03 12:59:55.979836: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcuda.dylib
  2019-09-03 12:59:55.986792: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:59:55.986920: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1640] Found device 0 with properties: 
  name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.607
  pciBusID: 0000:43:00.0
  2019-09-03 12:59:55.987018: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudart.10.0.dylib
  2019-09-03 12:59:55.991541: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcublas.10.0.dylib
  2019-09-03 12:59:55.994239: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcufft.10.0.dylib
  2019-09-03 12:59:55.995071: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcurand.10.0.dylib
  2019-09-03 12:59:55.998959: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcusolver.10.0.  dylib
  2019-09-03 12:59:56.002247: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcusparse.10.0.  dylib
  2019-09-03 12:59:56.007425: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudnn.7.dylib
  2019-09-03 12:59:56.007535: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:59:56.007865: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:59:56.008106: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1763] Adding visible gpu devices: 0
  2019-09-03 12:59:56.008206: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudart.10.0.dylib
  2019-09-03 12:59:56.644317: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1181] Device interconnect StreamExecutor with strength 1 edge matrix:
  2019-09-03 12:59:56.644331: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1187]      0 
  2019-09-03 12:59:56.644335: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1200] 0:   N 
  2019-09-03 12:59:56.644505: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:59:56.644706: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:59:56.644874: I tensorflow/stream_executor/cuda/cuda_gpu_executor  .cc:966] OS X does not support NUMA - returning NUMA node zero
  2019-09-03 12:59:56.644990: I tensorflow/core/common_runtime/gpu/gpu_device.cc:  1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU  :0 with 9011 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080   Ti, pci bus id: 0000:43:00.0, compute capability: 6.1)
  WARNING:tensorflow:From /Users/alex/Documents/workon/tf_gpu_test/gpu_test3.py:  34: The name tf.keras.backend.set_session is deprecated. Please use tf.  compat.v1.keras.backend.set_session instead.
  
  WARNING:tensorflow:From /Users/alex/anaconda3/envs/tf_gpu/lib/python3.7/site-  packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.  __init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and   will be removed in a future version.
  Instructions for updating:
  Call initializer instance with the dtype argument instead of passing it to the   constructor
  Epoch 1/20
  2019-09-03 12:59:57.017863: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcublas.10.0.dylib
  2019-09-03 12:59:57.227410: I tensorflow/stream_executor/platform/default/  dso_loader.cc:42] Successfully opened dynamic library libcudnn.7.dylib
  60000/60000 [==============================] - 7s 112us/sample - loss: 0.1333 -   acc: 0.9602
  Epoch 2/20
  60000/60000 [==============================] - 5s 88us/sample - loss: 0.0456 -   acc: 0.9856
  Epoch 3/20
  60000/60000 [==============================] - 5s 90us/sample - loss: 0.0273 -   acc: 0.9909
  Epoch 4/20
  60000/60000 [==============================] - 5s 86us/sample - loss: 0.0182 -   acc: 0.9940
  Epoch 5/20
  60000/60000 [==============================] - 5s 85us/sample - loss: 0.0131 -   acc: 0.9956
  Epoch 6/20
  60000/60000 [==============================] - 5s 83us/sample - loss: 0.0090 -   acc: 0.9968
  Epoch 7/20
  59680/60000 [============================>.] - ETA: 0s - loss: 0.0052 - acc: 0.  9985
  Stop as accuracy on training data is over 99.8%
  60000/60000 [==============================] - 5s 82us/sample - loss: 0.0052 -   acc: 0.9985
  Training Duration 37.847 sec
  10000/10000 [==============================] - 0s 43us/sample - loss: 0.0509 -   acc: 0.9865
  
  Process finished with exit code 0
  ```

  - Using the eGPU, it takes ~5s per epoch. This is approximately 3x faster than using the CPU (~16s/epoch). It seems less of a bump than I had expected;
  - Same model running in colab gives approximately 6x bump (~10s/epoch with GPU; ~60s/epoch with CPU).
</details>

<details>
  <summary>cuda-smi</summary>

  - Download and complie by running below commands in terminal:
  ```bash
  $ git clone https://github.com/phvu/cuda-smi
  $ cd cuda-smi
  $ ./compile.sh
  $ sudo scp cuda-smi /usr/local/cuda/bin/
  $ sudo chmod 755 /usr/local/cuda/bin/cuda-smi
  ```
  - Run below command in terminal:
  ```bash
  $ cuda-smi
  Device 0 [PCIe 0:67:0.0]: GeForce GTX 1080 Ti (CC 6.1): 9241.9 of 11264 MB (i.e. 82%) Free
  ```
</details>


[101]:    https://github.com/quant4ds/tensorflow_gpu_macOS#motivation
[102]:    https://github.com/quant4ds/tensorflow_gpu_macOS#environment
[103]:    https://github.com/quant4ds/tensorflow_gpu_macOS#hardware
[104]:    https://github.com/quant4ds/tensorflow_gpu_macOS#backup
[105]:    https://github.com/quant4ds/tensorflow_gpu_macOS#macos
[106]:    https://github.com/quant4ds/tensorflow_gpu_macOS#nvidia-egpu-installation
[107]:    https://github.com/quant4ds/tensorflow_gpu_macOS#egpu
[108]:    https://github.com/quant4ds/tensorflow_gpu_macOS#xcode
[109]:    https://github.com/quant4ds/tensorflow_gpu_macOS#cuda
[110]:    https://github.com/quant4ds/tensorflow_gpu_macOS#cudnn
[111]:    https://github.com/quant4ds/tensorflow_gpu_macOS#optional-nccl
[112]:    https://github.com/quant4ds/tensorflow_gpu_macOS#egpu-test
[113]:    https://github.com/quant4ds/tensorflow_gpu_macOS#bootcamp-windows
[114]:    https://github.com/quant4ds/tensorflow_gpu_macOS#tensorflow-gpu
[117]:    https://github.com/quant4ds/tensorflow_gpu_macOS#self-compilation
[118]:    https://github.com/quant4ds/tensorflow_gpu_macOS#tensorflow-test
