# Tensorflow 1.14 on macOS with Nvidia GPU Support
(by quant4ds; 9/1/2019)
<!-- OSX10.13.6-17G2208_Xcode9.4.1_tensorflow1.14_python3.7.3_CUDA10.0_cuDNN7.4.2 -->

## Table of Contents
- [Motivation][101]
- [Environment][102]
- [Hardware][103]
- [BACKUP][104]
- [macOS Preparation][105]
- [Nvidia eGPU Installation][106]
    + [Xcode][107]
    + [purge-wrangler][108]
    + [CUDA][109]
    + [cuDNN][110]
    + [(optional) NCCL][111]
- [eGPU Test][112]
- [Bootcamp Windows][113]
- [Tensorflow-GPU][114]
    + [Python][115]
    + [Pre-compiled Wheel][116]
    + [Self-compilation][117]
- [Tensorflow Test][118]

## Motivation
The unfortunate battle between Apple and Nvidia sets obstacles for Mac users to utilize majority of Nvidia GPU with their machines. What exacerbates our lives is the announcement from tensorflow team to stop support macOS after v1.2. 

It's been more than a year since the announcement of Mojave in 2018. The web driver for their GPUs under the new macOS has not been published by Nvidia. And it's unknown when, if at all, such driver would be available for 10.14 or 10.15. 

Meanwhile, however, effortless work has been done in the unofficial way by multiple communities and individuals. Their work has succefully bridged the gaps to some level. After extensive amount of research and experiment, I am able aggregate necessary information to benefit from their generous contributions. This guide is to take notes of my experience in the complete process to connect a Nvidia GeForce GTX card as an eGPU to work with my MBP, and enable tensorflow to leverage the eGPU in machine learning. 

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
<span style="color:red"> **Back up!!** </span>

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
      
</details>

- Download macOS 10.13.6
    + **For 2018 and 2019 MBPs with touch bar, you have to download a special built of the High Sierra (i.e., 17G2208).** 
        * It's reported that regular build High Sierra (i.e., 17G65) is not compatible with 2018 MBP with touch bar. I tested on my 2019 MBP and it's also no-go. 
    + Four ways to download 
        * Direct download link if you can find one - I used this [17G2208](http://oceanofdmg.com/download-macos-high-sierra-v10-13-6-17g2208-app-store-dmg/); 
        * Use the [patcher](http://dosdude1.com/highsierra/) (only for regular build 17G65);
        * First directly download the compact downloader from [app store](https://apps.apple.com/us/app/macos-high-sierra/id1246284741?mt=12), followed by modifying the file as shown in [这里](https://www.newlearner.site/2019/07/22/full-size-macos.html) (only for regular build 17G65); and
        * Download [installinstallmacos.py](https://github.com/munki/macadmin-scripts/blob/master/installinstallmacos.py) and follow the [instruction](https://github.com/munki/macadmin-scripts/blob/master/docs/installinstallmacos.md) (for all latest versions including both 17G2208 and 17G65). 
- Create bootable flash drive
    + **The flash drive should be at least 8gb in capacity.**
    + 

## Nvidia eGPU Installation
> ### Xcode
> ### purge-wrangler
> ### CUDA
> ### cuDNN
> ### (optional) NCCL
## eGPU Test
## Bootcamp Windows
## Tensorflow-GPU
> ### Python
> ### Pre-compiled Wheel
> ### Self-compilation
## Tensorflow Test



[101]:    https://github.com/quant4ds/tensorflow_gpu_macOS#purpose
[102]:    https://github.com/quant4ds/tensorflow_gpu_macOS#environment
[103]:    https://github.com/quant4ds/tensorflow_gpu_macOS#hardware
[104]:    https://github.com/quant4ds/tensorflow_gpu_macOS#backup
[105]:    https://github.com/quant4ds/tensorflow_gpu_macOS#macos
[106]:    https://github.com/quant4ds/tensorflow_gpu_macOS#nvidia-egpu-installation
[107]:    https://github.com/quant4ds/tensorflow_gpu_macOS#xcode
[108]:    https://github.com/quant4ds/tensorflow_gpu_macOS#purge-wrangler
[109]:    https://github.com/quant4ds/tensorflow_gpu_macOS#cuda
[110]:    https://github.com/quant4ds/tensorflow_gpu_macOS#cudnn
[111]:    https://github.com/quant4ds/tensorflow_gpu_macOS#optional-nccl
[112]:    https://github.com/quant4ds/tensorflow_gpu_macOS#egpu-test
[113]:    https://github.com/quant4ds/tensorflow_gpu_macOS#bootcamp-windows
[114]:    https://github.com/quant4ds/tensorflow_gpu_macOS#tensorflow-gpu
[115]:    https://github.com/quant4ds/tensorflow_gpu_macOS#python
[116]:    https://github.com/quant4ds/tensorflow_gpu_macOS#pre-compiled-wheel
[117]:    https://github.com/quant4ds/tensorflow_gpu_macOS#self-compilation
[118]:    https://github.com/quant4ds/tensorflow_gpu_macOS#tensorflow-test
