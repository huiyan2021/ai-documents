# Get Started with Intel MLPerf v3.0 Submission with Intel Optimized Docker Images

MLPerf is a benchmark for measuring the performance of machine learning
systems. It provides a set of performance metrics for a variety of machine
learning tasks, including image classification, object detection, machine
translation, and others. The benchmark is representative of real-world
workloads and as a fair and useful way to compare the performance of different
machine learning systems.


In this document, we'll show how to run Intel MLPerf v3.0 submission with Intel
optimized Docker images.

## Get the latest MLPerf 3.0 release

Please follow the below commands to get the latest mlperf 3.0 release.
```
git clone https://github.com/mlcommons/inference_results_v3.0.git
cd inference_results_v3.0
wget https://raw.githubusercontent.com/intel-ai-tce/ai-documents/mlperf_patches/AEM/mlperf/patches/0001-updates-for-3.0-submission.patch
git am 0001-updates-for-3.0-submission.patch
```

## Intel Docker Images for MLPerf

The Intel optimized Docker images for MLPerf v3.0 can be built using the
Dockerfiles.   
__Please refer to "Build & Run Docker container from Dockerfile" sub-section in each model section.__

Example for building docker image with Dockerfile:
```
cd inference_results_v3.0/closed/Intel/code/resnet50/pytorch-cpu/docker/

bash build_resnet50_contanier.sh
```

## Validated HW configuration:

| System Info     | Configuration detail                 |
| --------------- | ------------------------------------ |
| CPU             | SPR                       |
| OS              | CentOS  Stream 8                     |
| Kernel          | 6.1.11-1.el8.elrepo.x86_64 |
| Memory          | 1024GB (16x64GB 4800MT/s [4800MT/s]) |
| Disk            | 1TB NVMe                             |

## Recommmended BIOS Knobs:

| BIOS Knobs     | Recommended Value                 |
| --------------- | ------------------------------------ |
| Hyperthreading  | Enabled                              |
| Turbo Boost|  Enabled                                |
| Core Prefetchers         |      Hardware,Adjacent Cache,DCU Streamer,DCU IP                              |
| LLC Prefetch    |    Disable                            |
| CPU Power and Perf Policy | Performance |
| NUMA-based Cluster | Disabled |
| Energy Perf Bias | Performance |
| Energy Efficient Turbo | Disabled |

Please also refer to [Eagle Stream Platform Performance & Power Optimization Guide](https://cdrdv2.intel.com/v1/dl/getContent/733546?explicitVersion=true) for more details.

## Check System Health Using Intel® System Health Inspector:
Intel® System Health Inspector (aka svr-info) is a Linux OS utility for assessing the state and health of Intel Xeon computers. It is suggested to use svr-info first to check any system configuration issue before running any benchmark. Follow [the Quick Start Guide](https://github.com/intel/svr-info#quick-start) for downloading and installation. The following are several key factors effecting the model performance.

<details>
<summary> CPU </summary>
Couple CPU features impact MLPerf performance via related BIOS knobs, so please double check the CPU features with your BIOS knobs.
Some important CPU features are Hyperthreading, number of NUMA nodes, Prefetchers and Intel Turbo Boost.
<br><img src="BIOS_examples/CPU_setting.png" width="300" height="600"><br>
</details>

<details>
<summary> Memory </summary>
One important system configuration is balanced DIMM population, which is suggested to set as balanced to get optimized performance. <br> 
Populate as many channels per socket as possible prior to adding additional DIMMs to the channel.   
It might impact the memory bandwidth if two dimm share one channel. <br>   
Please also refer to Chapter 4 in <a href="https://cdrdv2.intel.com/v1/dl/getContent/733546?explicitVersion=true">Eagle Stream Platform Performance & Power Optimization Guide</a> for more details.  <br> 
     
From the results of svr-info, an example of unbalanced DIMM population is shown as follows,
<br><img src="BIOS_examples/Unbalanced_DIMM.png" width="300" height="600"><br>

An exmaple of Balanced DIMM population is shown as follows,     
<br><img src="BIOS_examples/Balanced_DIMM.png" width="300" height="600"><br>

You should also see good numbers for memory NUMA bandwidth if you also benchmark memory via svr-info. <br>
Here are some reference numbers from a 2S SPR system.
<br><img src="BIOS_examples/mem_bandwidth.png" width="200" height="150"><br>     
     
</details>

<details>
<summary> Power  </summary>
We recommend the intel_pstate Frequency Driver. <br>
For best performance, set the Frequency Governor and Power and Perf Policy to performance. <br>
Here are related recommended power settings from svr-info. 
<br><img src="BIOS_examples/power_setting.png" width="400" height="300"><br>
</details>


## Best Known Configurations:

```
sudo bash run_clean.sh
```

## Running models:
In the following sections, we'll show you how to set up and run each of the six models:

* [3DUNET](#get-started-with-3dunet)
* [BERT](#get-started-with-bert)
* [DLRM](#get-started-with-dlrm)
* [RESNET50](#get-started-with-resnet50)
* [RETINANET](#get-started-with-retinanet)
* [RNNT](#get-started-with-rnnt)

---


## Get Started with 3DUNET
### Build & Run Docker container from Dockerfile
If you haven't already done so, build the Intel optimized Docker image for 3DUNET using:
```
cd inference_results_v3.0/closed/Intel/code/3d-unet-99.9/pytorch-cpu/docker
bash build_3dunet_container.sh
```

### Prerequisites
Use these commands to prepare the 3DUNET dataset and model on your host system:

```
mkdir 3dunet
cd 3dunet
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
cd ..
```

### Set Up Environment
Follow these steps to set up the docker instance and preprocess the data.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we pulled earlier.
Replace ``/path/of/3dunet`` with the 3dunet folder path created earlier:
```
docker run --name intel_3dunet --privileged -itd -v /path/to/3dunet:/root/mlperf_data/3dunet-kits --net=host --ipc=host mlperf_inference_3dunet:3.0
```

#### Login to Docker Instance
Login into a bashrc shell in the Docker instance.
```
docker exec -it intel_3dunet bash
```

#### Preprocess Data
If you need a proxy to access the internet, replace ``your host proxy`` with
the proxy server for your environment.  If no proxy is needed, you can skip
this step:

```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

Preprocess the data and download the model using the provided script:
```
pip install numpy==1.23.5
cd code/3d-unet-99.9/pytorch-cpu/
bash process_data_model.sh 
```

### Run the Benchmark

```
# 3dunet only has offline mode
bash run.sh perf # offline performance
bash run.sh acc  # offline accuracy
```

### Get the Results

* Check log file. Performance results are in ``./output/mlperf_log_summary.txt``.
  Verify that you see ``results is: valid``.

* For offline mode performance, check the field ``Samples per second:``
* Accuracy results are in ``./output/accuracy.txt``.  Check the field ``mean =``.
* The performance result is controled by the value of "target_qps" in user_<number of sockets>_socket.conf file. The scripts will automatically select user_<number of sockets>_socket.conf file according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user_<number of sockets>_socket.conf files.

Save these output log files elsewhere when each test is completed as
they will be overwritten by the next test.


##  Get started with BERT
The docker container can be created either by building it using the Dockerfile or pulling the image from Dockerhub (if available).

### Build & Run Docker container from Dockerfile
If you haven't already done so, build and run the Intel optimized Docker image for BERT using:
```
cd inference_results_v3.0/closed/Intel/code/bert-99/pytorch-cpu/docker/

bash build_bert-99_contanier.sh
```

### Prerequisites
Use these commands to prepare the BERT dataset and model on your host system:

```
cd /data/mlperf_data   # or path to where you want to store the data
mkdir bert
cd bert
mkdir dataset
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O dataset/dev-v1.1.json
git clone https://huggingface.co/bert-large-uncased model
cd model
wget https://zenodo.org/record/4792496/files/pytorch_model.bin?download=1 -O pytorch_model.bin
```

### Set Up Environment
Follow these steps to set up the docker instance and preprocess the data.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we pulled or built earlier.
Replace /path/of/bert with the bert folder path created earlier (i.e. /data/mlperf_data/bert):

```
docker run --name bert_3-0 --privileged -itd --net=host --ipc=host \
  -v /path/of/bert:/data/mlperf_data/bert <bert docker image ID>
```

#### Login to Docker Instance
Login into a bashrc shell in the Docker instance.
```
docker exec -it bert_3-0 bash
```

#### Convert Dataset and Model
If you need a proxy to access the internet, replace ``your host proxy`` with
the proxy server for your environment.  If no proxy is needed, you can skip
this step:

```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

```
cd code/bert-99/pytorch-cpu
export DATA_PATH=/data/mlperf_data/bert
bash convert.sh
```

### Run the Benchmark

```
bash run.sh                    #offline performance
bash run.sh --accuracy         #offline accuracy
bash run_server.sh             #server performance
bash run_server.sh --accuracy  #server accuracy
```


### Get the Results

Check the performance log file ``./test_log/mlperf_log_summary.txt``:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``
* The performance results are controled by the value of "target_qps" in user_<number of sockets>_socket.conf file. The scripts will automatically select user_<number of sockets>_socket.conf file according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user_<number of sockets>_socket.conf files.

Check the accuracy log file ``./test_log/accuracy.txt``.

* Check the field ``f1``


Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

---

## Get started with DLRM
### Build & Run Docker container from Dockerfile
If you haven't already done so, build the Intel optimized Docker image for DLRM using:
```
# Please get compiler first.
cd inference_results_v3.0/closed/Intel/code/dlrm-99.9
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18679/l_HPCKit_p_2022.2.0.191.sh

# Build docker image
cd inference_results_v3.0/closed/Intel/code/dlrm-99.9/pytorch-cpu/docker
bash build_dlrm-99.9_container.sh
```

### Prerequisites
Use these commands to prepare the Deep Learning Recommendation Model (DLRM)
dataset and model on your host system:

```
cd /data/   # or path to where you want to store the data
mkdir -p /data/dlrm/model
mkdir -p /data/dlrm/terabyte_input

# download dataset
# Create a directory (such as /data/dlrm/terabyte_input) which contain:
#	    day_fea_count.npz
#	    terabyte_processed_test.bin
#
# Learn how to get the dataset from:
#     https://github.com/facebookresearch/dlrm
# You can also copy it using:
#     scp -r mlperf@10.112.230.156:/home/mlperf/dlrm_data/* /data/dlrm/terabyte_input
#
# download model
# Create a directory (such as /data/dlrm/model):
cd /data/dlrm/model
wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```

### Set Up Environment
Follow these steps to set up the docker instance.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we pulled earlier.
Replace ``/path/of/dlrm`` with the ``dlrm`` folder path created earlier (/data/dlrm for example):

```
docker run --name intel_inference_dlrm --privileged -itd --net=host --ipc=host \
  -v /path/of/dlrm:/data/mlperf_data/raw_dlrm mlperf_inference_dlrm:3.0
```

#### Login to Docker Container
Login into a bashrc shell in the Docker instance.

```
docker exec -it intel_inference_dlrm bash
```

### Preprocess model and dataset

If you need a proxy to access the internet, replace ``your host proxy`` with
the proxy server for your environment.  If no proxy is needed, you can skip
this step:

```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

```
cd /opt/workdir/code/dlrm/pytorch-cpu
export MODEL=/data/mlperf_data/raw_dlrm/model
export DATASET=/data/mlperf_data/raw_dlrm/terabyte_input
export DUMP_PATH=/data/mlperf_data/dlrm
bash dump_model_dataset.sh
```

### Run the Benchmark

```
export MODEL_DIR=/data/mlperf_data/dlrm
export DATA_DIR=/data/mlperf_data/dlrm

bash runcppsut                     # offline performance
bash runcppsut accuracy	           # offline accuracy
bash runcppsut performance server  # server performance
bash runcppsut accuracy server     # server accuracy
```

### Get the Results

Check the appropriate offline or server performance log file, either
``./output/PerformanceOnly/Offline/mlperf_log_summary.txt`` or
``./output/PerformanceOnly/Server/mlperf_log_summary.txt``:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``
* The performance result is controled by the value of "target_qps" in user_<number of sockets>_socket.conf file. The scripts will automatically select user_<number of sockets>_socket.conf file according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user_<number of sockets>_socket.conf files.

Check the appropriate offline or server accuracy log file, either
``./output/AccuracyOnly/Offline/accuracy.txt`` or
``./output/AccuracyOnly/Server/accuracy.txt``:

* Check the field ``AUC``

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

---

##  Get Started with ResNet50
The docker container can be created either by building it using the Dockerfile or pulling the image from Dockerhub (if available). Please download the Imagenet dataset on the host system before starting the container.

### Download Imagenet Dataset for Calibration
Download ImageNet (50000) dataset
```
bash download_imagenet.sh
```

### Build & Run Docker container from Dockerfile
If you haven't already done so, build and run the Intel optimized Docker image for ResNet50 using:
```
cd inference_results_v3.0/closed/Intel/code/resnet50/pytorch-cpu/docker/

bash build_resnet50_contanier.sh

docker run -v </path/to/ILSVRC2012_img_val>:/opt/workdir/code/resnet50/pytorch-cpu/ILSVRC2012_img_val -it --privileged <resnet docker image ID> /bin/bash

cd code/resnet50/pytorch-cpu
```

### Prepare Calibration Dataset & Download Model ( Inside Container )
If you need a proxy to access the internet, replace your host proxy with the proxy server for your environment. If no proxy is needed, you can skip this step:
```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

Prepare calibration 500 images into folders
```
cd /opt/workdir/code/resnet50/pytorch-cpu
bash prepare_calibration_dataset.sh
```

Download the model
```
bash download_model.sh
```
The downloaded model will be saved as ```resnet50-fp32-model.pth```

### Quantize Torchscript Model and Check Accuracy 
+ Set the following paths:
```
export DATA_CAL_DIR=calibration_dataset
export CHECKPOINT=resnet50-fp32-model.pth
```
+ Generate scales and models
```
bash generate_torch_model.sh
```

The *start* and *end* parts of the model are also saved (respectively named) in ```models```


### Run Benchmark (Common for Docker & Baremetal)

```
export DATA_DIR=${PWD}/ILSVRC2012_img_val
export RN50_START=models/resnet50-start-int8-model.pth
export RN50_END=models/resnet50-end-int8-model.pth
export RN50_FULL=models/resnet50-full.pth
```

#### Performance
+ Offline
```
bash run_offline.sh <batch_size>
```

+ Server
```
bash run_server.sh
```

#### Accuracy
+ Offline
```
bash run_offline_accuracy.sh <batch_size>
```

+ Server
```
bash run_server_accuracy.sh
```


### Get the Results

Check the ``./mlperf_log_summary.txt`` log file:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``
* The performance result is controled by the value of "target_qps" in ./src/user_<number of sockets>_socket.conf file. The scripts will automatically select user_<number of sockets>_socket.conf file according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user_<number of sockets>_socket.conf files.
     
     
Check the ``./offline_accuracy.txt`` or ``./server_accuracy.txt`` log file:

* Check the field ``accuracy``

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

---

##  Get Started with Retinanet

The docker container can be created either by building it using the Dockerfile or pulling the image from Dockerhub (if available). Please download the Imagenet dataset on the host system before starting the container.

### Download the dataset
+ Install dependencies (**python3.9 or above**)
```
pip3 install --upgrade pip --user
pip3 install opencv-python-headless==4.5.3.56 pycocotools==clear2.0.2 fiftyone==0.16.5
```

+ Setup env vars
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}

export ENV_DEPS_DIR=${CUR_DIR}/retinanet-env
```

+ Download OpenImages (264) dataset
```
bash openimages_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages
```
Images are downloaded to `${WORKLOAD_DATA}/openimages`

+ Download Calibration images
```
bash openimages_calibration_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages-calibration
```
Calibration dataset downloaded to `${WORKLOAD_DATA}/openimages-calibration`

Note: If you meet any obstacles on downloading the dataset, please try again in the docker container to be launched after [Build & Run Docker container from Dockerfile](Build & Run Docker container from Dockerfile).

### Download Model
```
wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O 'retinanet-model.pth'
mv 'retinanet-model.pth' ${WORKLOAD_DATA}/
```

### Build & Run Docker container from Dockerfile
If you haven't already done so, build and run the Intel optimized Docker image for Retinanet using:
```
cd inference_results_v3.0/closed/Intel/code/retinanet/pytorch-cpu/docker/

bash build_retinanet_contanier.sh

docker run --name intel_retinanet --privileged -itd --net=host --ipc=host -v ${WORKLOAD_DATA}:/opt/workdir/code/retinanet/pytorch-cpu/data <retinanet docker image ID> 

docker exec -it intel_retinanet bash 

cd code/retinanet/pytorch-cpu/
```

### Calibrate and generate torchscript model

If you need a proxy to access the internet, replace your host proxy with the proxy server for your environment. If no proxy is needed, you can skip this step:
```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

Run Calibration
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
export MODEL_CHECKPOINT=${WORKLOAD_DATA}/retinanet-model.pth
export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json

cd /opt/workdir/code/retinanet/pytorch-cpu/retinanet-env/vision
git checkout 8e078971b8aebdeb1746fea58851e3754f103053
python setup.py install && python setup.py develop

cd /opt/workdir/code/retinanet/pytorch-cpu
bash run_calibration.sh
```

### Set Up Environment

Export the environment settings
```
source setup_env.sh
```

### Run the Benchmark

```

# Run one of these performance or accuracy scripts at a time
# since the log files will be overwritten on each run

# for offline performance
bash run_offline.sh

# for server performance
bash run_server.sh

# for offline accuracy
bash run_offline_accuracy.sh

# for server accuracy
bash run_server_accuracy.sh
```


### Get the results

Check the ``./mlperf_log_summary.txt`` log file:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``
* The performance result is controled by the value of "target_qps" in user_<number of sockets>_socket.conf file. The scripts will automatically select user_<number of sockets>_socket.conf file according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user_<number of sockets>_socket.conf files.
     
Check the ``./accuracy.txt`` log file:

* Check the field ``mAP``

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

## Get Started with RNNT

### Build & Run Docker container from Dockerfile
If you haven't already done so, build the Intel optimized Docker image for RNNT using:
```
cd inference_results_v3.0/closed/Intel/code/rnnt/pytorch-cpu/docker/
bash build_rnnt-99_container.sh
```

### Set Up Environment
Follow these steps to set up the docker instance.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we built earlier.
```
docker run --name intel_rnnt --privileged -itd -v /data/mlperf_data:/data/mlperf_data \
--net=host --ipc=host mlperf_inference_rnnt:3.0
```

#### Login to Docker Container
Get the Docker container ID and login into a bashrc shell in the Docker instance using ``docker exec``.

```
docker ps -a #get container "id"
docker exec -it <id> bash
cd /opt/workdir/code/rnnt/pytorch-cpu
```

+ Setup env vars

```
export LD_LIBRARY_PATH=/opt/workdir/code/rnnt/pytorch-cpu/third_party/lib:$LD_LIBRARY_PATH
```

If you need a proxy to access the internet, replace your host proxy with the proxy server for your environment. If no proxy is needed, you can skip this step:
```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

### Run the Benchmark

The provided ``run.sh`` script abstracts the end-to-end process for RNNT:
| STAGE | STEP  |
| ------- | --- | 
| 0 | Download model |
| 1 | Download dataset |
| 2 | Pre-process dataset |
| 3 | Calibration |
| 4 | Build model |
| 5 | Run Offline/Server accuracy & benchmark |

Run ``run.sh`` with ``STAGE=0`` to invoke all the steps requried to run the benchmark (i.e download the model & dataset, preprocess the data, calibrate and build the model):

```
 SKIP_BUILD=1 STAGE=0 bash run.sh
```
or to skip to stage 5 without previous steps: Offline/Server accuracy and benchmark:
```
 SKIP_BUILD=1 STAGE=5 bash run.sh
```

### Get the Results

Check the appropriate offline or server performance log files, either
``./logs/Server/performance/.../mlperf_log_summary.txt`` or
``./logs/Offline/performance/.../mlperf_log_summary.txt``:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``
* The performance result is controled by the value of "target_qps" in ./configs/user_<number of sockets>_socket.conf file. The scripts will automatically select user_<number of sockets>_socket.conf file according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user_<number of sockets>_socket.conf files.
     
Check the appropriate offline or server accuracy log file, either
``./logs/Server/accuracy/.../mlperf_log_summary.txt`` or
``./logs/Offline/accuracy/.../mlperf_log_summary.txt``:

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

