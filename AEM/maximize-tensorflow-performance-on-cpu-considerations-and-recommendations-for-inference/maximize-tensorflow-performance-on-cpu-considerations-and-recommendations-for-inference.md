To fully utilize the power of Intel® architecture (IA) for high performance, you can enable TensorFlow* to be powered by Intel’s highly optimized math routines in the Intel® oneAPI Deep Neural Network Library (oneDNN). oneDNN includes convolution, normalization, activation, inner product, and other primitives.

The oneAPI Deep Neural Network Library (oneDNN) optimizations are now available both in the official x86-64 TensorFlow and  Intel® Optimization for TensorFlow* after v2.5. Users can enable those CPU optimizations by setting the the environment variable **TF\_ENABLE\_ONEDNN\_OPTS=1** for the official x86-64 TensorFlow after v2.5.

Most of the recommendations work on both official x86-64 TensorFlow and  Intel® Optimization for TensorFlow. Some recommendations such as OpenMP tuning only applies to Intel® Optimization for TensorFlow.

For setting up Intel® Optimization for TensorFlow* framework, please refer to this [installation guide](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html).

## Maximum Throughput vs. Real-time Inference
You can perform deep learning inference using two different strategies, each with different performance measurements and recommendations. The first is Max Throughput (MxT), which aims to process as many images per second as possible, passing in batches of size > 1. For Max Throughput, you achieve better performance by exercising all the physical cores on a socket. With this strategy, you simply load up the CPU with as much work as you can and process as many images as you can in a parallel and vectorized fashion.

An altogether different strategy is Real-time Inference (RTI) where you typically process a single image as fast as possible. Here you aim to avoid penalties from excessive thread launching and orchestration among concurrent processes. The strategy is to confine and execute quickly. The best-known methods (BKMs) differ for these two strategies.

## TensorFlow Graph Options Improving Performance
Optimizing graphs help improve latency and throughput time by transforming graph nodes to have only inference related nodes and by removing all training nodes.

Users can use tools from TensorFlow github.  

**First, use freeze\_graph**

First, freezing the graph can provide additional performance benefits. The freeze\_graph tool, available as part of TensorFlow on GitHub, converts all the variable ops to const ops on the inference graph and outputs a frozen graph. With all weights frozen in the resulting inference graph, you can expect improved inference time. Here is a [LINK](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) to access the freeze\_graph tool.

**Second, Use optimize\_for\_inference**

When the trained model is used only for inference, after the graph has been frozen, additional transformations can help optimize the graph for inference. TensorFlow project on GitHub offers an easy to use optimization tool to improve the inference time by applying these transformations to a trained model output. The output will be an inference-optimized graph to improve inference time. Here is a [LINK](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py) to access the optimize\_for\_inference tool.

## TensorFlow Runtime Options Improving Performance
Runtime options heavily affect TensorFlow performance. Understanding them will help get the best performance out of the Intel Optimization of TensorFlow.

<details>
  <summary>intra\_/inter\_op\_parallelism\_threads</summary>
  <br>
  <b>Recommended settings (RTI):intra\_op\_parallelism = number of physical core per socket</b>
  <br><br>
  <b>Recommended settings: inter\_op\_parallelism = number of sockets</b>
  <br><br>
  <b>Users can put below bash commands into a bash script file, and then get the number of physical core per socket and number of sockets on your platform by executing the bash script file.</b>
  <br><br>
  <pre>
    total_cpu_cores=$(nproc)
    number_sockets=$(($(grep "^physical id" /proc/cpuinfo | awk '{print $4}' | sort -un | tail -1)+1))
    number_cpu_cores=$(( (total_cpu_cores/2) / number_sockets))
    <br>
    echo "number of CPU cores per socket: $number_cpu_cores";
    echo "number of socket: $number_sockets";
  </pre>
  <br>
  For example, here is how you can set the inter and intra\_op\_num\_threads by using <a href="https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks">TensorFlow Benchmark</a>.tf\_cnn\_benchmarks usage (shell)
  <br>
  <pre>python tf_cnn_benchmarks.py --num_intra_threads=&lt;number of physical cores per socket&gt; --num_inter_threads=&lt;number of sockets&gt;</pre>
  <b>intra\_op\_parallelism\_threads</b> and <b>inter\_op\_parallelism\_threads</b> are runtime variables defined in TensorFlow.
  <br><br>
  <b>ConfigProto</b>
  <br><br>
  The ConfigProto is used for configuration when creating a session. These two variables control number of cores to use.
  <br><br>
  <li>intra\_op\_parallelism\_threads</li>
  <br>
  This runtime setting controls parallelism inside an operation. For instance, if matrix multiplication or reduction is intended to be executed in several threads, this variable should be set. TensorFlow will schedule tasks in a thread pool that contains intra\_op\_parallelism\_threads threads. As illustrated later in Figure 2, OpenMP* threads are bound to thread context as close as possible on different cores. Setting this environment variable to the number of available physical cores is recommended.
  <br><br>
  <li>inter\_op\_parallelism\_threads</li>
  <br>
  NOTE: This setting is highly dependent on hardware and topologies, so it’s best to empirically confirm the best setting on your workload.
  <br><br>
  This runtime setting controls parallelism among independent operations. Since these operations are not relevant to each other, TensorFlow will try to run them concurrently in the thread pool that contains inter\_op\_parallelism\_threads threads. This variable should be set to the number of parallel paths where you want the code to run. For Intel® Optimization for TensorFlow, we recommend starting with the setting '2’, and adjusting after empirical testing.
</details>

<details>
  <summary>Data layout</summary>
  <br>
  <b>Recommended settings → data\_format = NHWC</b>
  <br>
  tf\_cnn\_benchmarks usage (shell)
  <br>
  <pre>python tf_cnn_benchmarks.py --num_intra_threads=&lt;number of physical cores per socket&gt; --num_inter_threads=&lt;number of sockets&gt; --data_format=NHWC</pre>
  <br>
  Efficiently using cache and memory yields remarkable improvements in overall performance. A good memory access pattern minimizes extra cost for accessing data in memory and improves overall processing. Data layout, how data is stored and accessed, plays an important role in achieving these good memory access patterns. Data layout describes how multidimensional arrays are stored linearly in memory address space.

  In most cases, data layout is represented by four letters for a two-dimensional image:

  - N: Batch size, indicates number of images in a batch.
  - C: Channel, indicates number of channels in an image.
  - W: Width, indicates number of horizontal pixels of an image.
  - H: Height, indicates number of vertical pixels of an image.
  <br>
  The order of these four letters indicates how pixel data are stored in the one-dimensional memory space. For instance, NCHW indicates pixel data are stored as width first, then height, then channel, and finally batch (Illustrated in Figure 2). The data is then accessed from left-to-right with channel-first indexing. NCHW is the recommended data layout for using oneDNN, since this format is an efficient data layout for the CPU. TensorFlow uses NHWC as its default data layout, but it also supports NCHW.

  ![Data Formats for Deep Learning NHWC and NCHW](/content/dam/develop/external/us/en/images/data-layout-nchw-nhwc-804042.png) 

  Figure 1: Data Formats for Deep Learning NHWC and NCHW

  <b>NOTE :</b> Intel Optimized TensorFlow supports both plain data formats like NCHW/NHWC and also oneDNN blocked data format since version 2.4. Using blocked format might help on vectorization but might introduce some data reordering operations in TensorFlow.

  Users could enable/disable usage of oneDNN blocked data format in Tensorflow by TF\_ENABLE\_MKL\_NATIVE\_FORMAT environment variable. By exporting TF\_ENABLE\_MKL\_NATIVE\_FORMAT=0, TensorFlow will use oneDNN blocked data format instead. Please check [oneDNN memory format](https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html) for more information about oneDNN blocked data format.

  We recommend users to enable NATIVE\_FORMAT by below command to achieve good out-of-box performance.
  export TF\_ENABLE\_MKL\_NATIVE\_FORMAT=1 (or 0)
</details>

<details>
  <summary>oneDNN Related Runtime Environment Variables</summary>
  <br>
  There are some runtime arguments related to oneDNN optimizations in TensorFlow.
  <br>
  Users could tune those runtime arguments to achieve better performance.

  | Environment Variables | Default | Purpose |
  | --- | --- | --- |
  | TF\_ENABLE\_ONEDNN\_OPTS | True | Enable/Disable oneDNN optimization |
  | TF\_ONEDNN\_ASSUME\_FROZEN\_WEIGHTS | False | Frozen weights for inference.<br>Better inference performance is achieved with frozen graphs.<br>Related ops: fwd conv, fused matmul |
  | TF\_ONEDNN\_USE\_SYSTEM\_ALLOCATOR | False | Use system allocator or BFC allocator in MklCPUAllocator.<br>Usage:<br><li>Set it to true for better performance if the workload meets one of following conditions:</li><ul><li>small allocation.</li><li>inter\_op\_parallelism\_threads is large.</li><li>has a weight sharing session</li></ul><li>Set it to False to use large-size allocator (BFC).</li>In general, set this flag to True for inference, and set this flag to False for training. |
  | TF\_MKL\_ALLOC\_MAX\_BYTES | 64 | MklCPUAllocator: Set upper bound on memory allocation. Unit:GB|
  | TF\_MKL\_OPTIMIZE\_PRIMITIVE\_MEMUSE | True | Use oneDNN primitive caching or not.<li>Set False to enable primitive caching in TensorFlow.</li><li>Set True to disable primitive caching in TensorFlow and oneDNN might cache those primitives for TensorFlow.</li>Disabling primitive caching will reduce memory usage in TensorFlow but impacts performance.|
</details>

<details>
  <summary>Memory Allocator</summary>
  <br>
  For deep learning workloads, TCMalloc can get better performance by reusing memory as much as possible than default malloc funtion. <a href="https://google.github.io/tcmalloc/overview.html">TCMalloc</a> features a couple of optimizations to speed up program executions. TCMalloc is holding memory in caches to speed up access of commonly-used objects. Holding such caches even after deallocation also helps avoid costly system calls if such memory is later re-allocated. Use environment variable LD\_PRELOAD to take advantage of one of them.
  <br>
    <pre>
      $ sudo apt-get install google-perftools4
      $ LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python script.py ...
  </pre>
</details>

## Non-uniform memory access (NUMA) Controls Affecting Performance
<br>
NUMA, or non-uniform memory access, is a memory layout design used in data center machines meant to take advantage of locality of memory in multi-socket machines with multiple memory controllers and blocks. Running on a NUMA-enabled machine brings with it, special considerations. Intel® Optimization for TensorFlow runs inference workload best when confining both the execution and memory usage to a single NUMA node. When running on a NUMA-enabled system, recommendation is to set intra\_op\_parallelism\_threads to the numbers of local cores in each single NUMA-node.
<br><br>
Recommended settings: --cpunodebind=0 --membind=0
<br><br>
Usage (shell)
<br>
<pre>numactl --cpunodebind=0 --membind=0 python</pre>

<details>
  <summary>Concurrent Execution</summary>
  <br>
  You can optimize performance by breaking up your workload into multiple data shards and then running them concurrently on more than one NUMA node. On each node (N), run the following command:
  <br><br>
  Usage (shell)
  <br>
  <pre>numactl --cpunodebind=N --membind=N python</pre>
  For example, you can use the “&” command to launch simultaneous processes on multiple NUMA nodes:
  <br>
  <pre>numactl --cpunodebind=0 --membind=0 python & numactl --cpunodebind=1 --membind=1 python</pre>
  <br>
</details>

<details>
<summary>CPU Affinity</summary>
  <br>
  Users could bind threads to specific CPUs via "--physcpubind=cpus" or "-C cpus"
  <br><br>
  Setting its value to <b>"0-N"</b> will bind  threads to physical cores 0 to N only.
  <br><br>
  Usage (shell)
  <pre>numactl --cpunodebind=N --membind=N -C 0-N python</pre>
  For example, you can use the “&” command to launch simultaneous processes on multiple NUMA nodes on physical CPU 0 to 3 and 4 to 7:
  <pre>numactl --cpunodebind=0 --membind=0 -C 0-3 python & numactl --cpunodebind=1 --membind=1 -C 4-7 python</pre>
  NOTE : oneDNN will <a href="https://github.com/oneapi-src/oneDNN/blob/e535ef2f8cbfbee4d385153befe508c6b054305e/src/cpu/platform.cpp#LL238">get the CPU affinity mask</a> from users' numactl setting and set the maximum number of working threads in the threadpool accordingly after TensorFlow v2.5.0 RC1.
</details>

