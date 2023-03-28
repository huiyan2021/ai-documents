To fully utilize the power of Intel® architecture (IA) for high performance, you can enable TensorFlow* to be powered by Intel’s highly optimized math routines in the Intel® oneAPI Deep Neural Network Library (oneDNN). oneDNN includes convolution, normalization, activation, inner product, and other primitives.

The oneAPI Deep Neural Network Library (oneDNN) optimizations are now available both in the official x86-64 TensorFlow and  Intel® Optimization for TensorFlow* after v2.5. Users can enable those CPU optimizations by setting the the environment variable **TF_ENABLE_ONEDNN_OPTS=1** for the official x86-64 TensorFlow after v2.5.

Most of the recommendations work on both official x86-64 TensorFlow and  Intel® Optimization for TensorFlow. Some recommendations such as OpenMP tuning only applies to Intel® Optimization for TensorFlow.

For setting up Intel® Optimization for TensorFlow* framework, please refer to this [installation guide](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html).

## Maximum Throughput vs. Real-time Inference
You can perform deep learning inference using two different strategies, each with different performance measurements and recommendations. The first is Max Throughput (MxT), which aims to process as many images per second as possible, passing in batches of size > 1. For Max Throughput, you achieve better performance by exercising all the physical cores on a socket. With this strategy, you simply load up the CPU with as much work as you can and process as many images as you can in a parallel and vectorized fashion.

An altogether different strategy is Real-time Inference (RTI) where you typically process a single image as fast as possible. Here you aim to avoid penalties from excessive thread launching and orchestration among concurrent processes. The strategy is to confine and execute quickly. The best-known methods (BKMs) differ for these two strategies.

## TensorFlow Graph Options Improving Performance
Optimizing graphs help improve latency and throughput time by transforming graph nodes to have only inference related nodes and by removing all training nodes.

Users can use tools from TensorFlow github.  

**First, use Freeze_graph**

First, freezing the graph can provide additional performance benefits. The freeze_graph tool, available as part of TensorFlow on GitHub, converts all the variable ops to const ops on the inference graph and outputs a frozen graph. With all weights frozen in the resulting inference graph, you can expect improved inference time. Here is a [LINK](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) to access the freeze_graph tool.

**Second, Use Optimize_for_inference**

When the trained model is used only for inference, after the graph has been frozen, additional transformations can help optimize the graph for inference. TensorFlow project on GitHub offers an easy to use optimization tool to improve the inference time by applying these transformations to a trained model output. The output will be an inference-optimized graph to improve inference time. Here is a [LINK](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py) to access the optimize_for_inference tool.

## TensorFlow Runtime Options Improving Performance
Runtime options heavily affect TensorFlow performance. Understanding them will help get the best performance out of the Intel Optimization of TensorFlow.

<details>
  <summary>intra_/inter_op_parallelism_threads</summary>
  <br>
  <b>Recommended settings (RTI):intra_op_parallelism = number of physical core per socket</b>
  <br><br>
  <b>Recommended settings: inter_op_parallelism = number of sockets</b>
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
  For example, here is how you can set the inter and intra_op_num_threads by using <a href="[https://www.runoob.com](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)">TensorFlow Benchmark</a>.tf_cnn_benchmarks usage (shell)
  <br>
  <pre>python tf_cnn_benchmarks.py --num_intra_threads=&lt;number of physical cores per socket&gt; --num_inter_threads=&lt;number of sockets&gt;</pre>
  <b>intra_op_parallelism_threads</b> and <b>inter_op_parallelism_threads</b> are runtime variables defined in TensorFlow.
  <br><br>
  <b>ConfigProto</b>
  <br><br>
  The ConfigProto is used for configuration when creating a session. These two variables control number of cores to use.
  <br><br>
  <li>intra_op_parallelism_threads</li>
  <br>
  This runtime setting controls parallelism inside an operation. For instance, if matrix multiplication or reduction is intended to be executed in several threads, this variable should be set. TensorFlow will schedule tasks in a thread pool that contains intra_op_parallelism_threads threads. As illustrated later in Figure 2, OpenMP* threads are bound to thread context as close as possible on different cores. Setting this environment variable to the number of available physical cores is recommended.
  <br><br>
  <li>inter_op_parallelism_threads</li>
  <br>
  NOTE: This setting is highly dependent on hardware and topologies, so it’s best to empirically confirm the best setting on your workload.
  <br><br>
  This runtime setting controls parallelism among independent operations. Since these operations are not relevant to each other, TensorFlow will try to run them concurrently in the thread pool that contains inter_op_parallelism_threads threads. This variable should be set to the number of parallel paths where you want the code to run. For Intel® Optimization for TensorFlow, we recommend starting with the setting '2’, and adjusting after empirical testing.
</details>

<details>
  <summary>Data layout</summary>
  <br>
  <b>Recommended settings → data_format = NHWC</b>
<br>
tf_cnn_benchmarks usage (shell)
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

![Data Formats for Deep Learning NHWC and NCHW](/content/dam/www/central-libraries/us/en/images/data-layout-nchw-nhwc-804042.png) 

Figure 1: Data Formats for Deep Learning NHWC and NCHW

NOTE : Intel Optimized TensorFlow supports both plain data formats like NCHW/NHWC and also oneDNN blocked data format since version 2.4. Using blocked format might help on vectorization but might introduce some data reordering operations in TensorFlow.

Users could enable/disable usage of oneDNN blocked data format in Tensorflow by TF_ENABLE_MKL_NATIVE_FORMAT environment variable. By exporting TF_ENABLE_MKL_NATIVE_FORMAT=0, TensorFlow will use oneDNN blocked data format instead. Please check [oneDNN memory format](https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html) for more information about oneDNN blocked data format.

We recommend users to enable NATIVE_FORMAT by below command to achieve good out-of-box performance.
export TF_ENABLE_MKL_NATIVE_FORMAT=1 (or 0)
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
For deep learning workloads, TCMalloc can get better performance by reusing memory as much as possible than default malloc funtion. <a href="https://google.github.io/tcmalloc/overview.html">TCMalloc</a> features a couple of optimizations to speed up program executions. TCMalloc is holding memory in caches to speed up access of commonly-used objects. Holding such caches even after deallocation also helps avoid costly system calls if such memory is later re-allocated. Use environment variable LD_PRELOAD to take advantage of one of them.
<br>
  <pre>
    $ sudo apt-get install google-perftools4
    $ LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python script.py ...
</pre>
</details>

## Non-uniform memory access (NUMA) Controls Affecting Performance
<br>
NUMA, or non-uniform memory access, is a memory layout design used in data center machines meant to take advantage of locality of memory in multi-socket machines with multiple memory controllers and blocks. Running on a NUMA-enabled machine brings with it, special considerations. Intel® Optimization for TensorFlow runs inference workload best when confining both the execution and memory usage to a single NUMA node. When running on a NUMA-enabled system, recommendation is to set intra_op_parallelism_threads to the numbers of local cores in each single NUMA-node.
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
  Setting its value to "0-N" will bind  threads to physical cores 0 to N only.
  <br><br>
  Usage (shell)
  <pre>numactl --cpunodebind=N --membind=N -C 0-N python</pre>
  For example, you can use the “&” command to launch simultaneous processes on multiple NUMA nodes on physical CPU 0 to 3 and 4 to 7:
  <pre>numactl --cpunodebind=0 --membind=0 -C 0-3 python & numactl --cpunodebind=1 --membind=1 -C 4-7 python</pre>
  NOTE : oneDNN will <a href="https://github.com/oneapi-src/oneDNN/blob/e535ef2f8cbfbee4d385153befe508c6b054305e/src/cpu/platform.cpp#LL238">get the CPU affinity mask</a> from users' numactl setting and set the maximum number of working threads in the threadpool accordingly after TensorFlow v2.5.0 RC1.
</details>

## OpenMP Technical Performance Considerations for Intel® Optimization for TensorFlow
<b>Note: This section is only for Intel® Optimization for TensorFlow, and it does not apply to official TensorFlow release.</b>  
<br>
Intel® Optimization for TensorFlow utilizes OpenMP to parallelize deep learnng model execution among CPU cores.
<br><br>
Users can use the following environment variables to be able to tune Intel® optimized TensorFlow performance . Thus, changing values of these environment variables affects performance of the framework. These environment variables will be described in detail in the following sections. We highly recommend users tuning these values for their specific neural network model and platform.
<br><br>
<details>
  <summary>OMP_NUM_THREADS</summary>
  <br>
  Recommended settings for CNN→ OMP_NUM_THREADS = num physical cores
  <br><br>
  Usage (shell)
  <br><br>
  <pre>export OMP_NUM_THREADS=num physical cores</pre>
  This environment variable sets the maximum number of threads to use for OpenMP parallel regions if no other value is specified in the application.
  <br><br>
  With Hyperthreading enabled, there are more than one hardware threads for a physical CPU core, but we recommend to use only one hardware thread for a physical CPU core to avoid cache miss problems. 
  <br><br>
  tf_cnn_benchmarks usage (shell)
  <br>
  <pre>OMP_NUM_THREADS=&lt;number of physical cores per socket&gt; python tf_cnn_benchmarks.py --num_intra_threads=&lt;number of physical cores per socket&gt; --num_inter_threads=&lt;number of sockets&gt; --data_format=NCHW</pre>

  Users can bind OpenMP threads to physical processing units. KMP_AFFINITY is used to take advantage of this functionality. It restricts execution of certain threads to a subset of the physical processing units in a multiprocessor computer.
<br><br>
The value can be a single integer, in which case it specifies the number of threads for all parallel regions. The value can also be a comma-separated list of integers, in which case each integer specifies the number of threads for a parallel region at a nesting level.
<br><br>
The first position in the list represents the outer-most parallel nesting level, the second position represents the next-inner parallel nesting level, and so on. At any level, the integer can be left out of the list. If the first integer in a list is left out, it implies the normal default value for threads is used at the outer-most level. If the integer is left out of any other level, the number of threads for that level is inherited from the previous level.
<br><br>
The default value is the number of logical processors visible to the operating system on which the program is executed. This value is recommended to be set to the number of physical cores.
</details>

<details>
  <summary>KMP_AFFINITY</summary>
  <br>
  <b>Recommended settings → KMP_AFFINITY=granularity=fine,verbose,compact,1,0</b>
  <pre>export KMP_AFFINITY=granularity=fine,compact,1,0</pre>
  tf_cnn_benchmarks usage (shell)
  <pre>OMP_NUM_THREADS=&lt;number of physical cores per socket&gt; python tf_cnn_benchmarks.py --num_intra_threads=&lt;number of physical cores per socket&gt; --num_inter_threads=&lt;number of sockets&gt; --data_format=NCHW --kmp_affinity=granularity=fine,compact,1,0</pre>
  
  Users can bind OpenMP threads to physical processing units. KMP_AFFINITY is used to take advantage of this functionality. It restricts execution of certain threads to a subset of the physical processing units in a multiprocessor computer.
<br><br>
Usage of this environment variable is as below.
<br><br>
KMP_AFFINITY=[,...][,][,]
<br><br>
Modifier is a string consisting of keyword and specifier. type is a string indicating the thread affinity to use. permute is a positive integer value, controls which levels are most significant when sorting the machine topology map. The value forces the mappings to make the specified number of most significant levels of the sort the least significant, and it inverts the order of significance. The root node of the tree is not considered a separate level for the sort operations. offset is a positive integer value, indicates the starting position for thread assignment. We will use the recommended setting of KMP_AFFINITY as an example to explain basic content of this environment variable.
<br><br>
KMP_AFFINITY=granularity=fine,verbose,compact,1,0
<br><br>
The modifier is granularity=fine,verbose. Fine causes each OpenMP thread to be bound to a single thread context. Verbose prints messages at runtime concerning the supported affinity, and this is optional. These messages include information about the number of packages, number of cores in each package, number of thread contexts for each core, and OpenMP thread bindings to physical thread contexts. Compact is value of type, assigning the OpenMP thread +1 to a free thread context as close as possible to the thread context where the OpenMP thread was placed.
<br><br>
NOTE The recommendation changes if Hyperthreading is disabled on your machine. In that case, the recommendation is:   KMP_AFFINITY=granularity=fine,verbose,compact if hyperthreading is disabled.
<br><br>
Fig. 2 shows the machine topology map when KMP_AFFINITY is set to these values. The OpenMP thread +1 is bound to a thread context as close as possible to OpenMP thread , but on a different core. Once each core has been assigned one OpenMP thread, the subsequent OpenMP threads are assigned to the available cores in the same order, but they are assigned on different thread contexts.
<br><br>
![OpenMP Global Thread Pool IDs](/content/dam/www/central-libraries/us/en/images/openmp-global-thread-pool-ids-804042.jpg)
<br> 
Figure 2. Machine topology map with setting KMP_AFFINITY=granularity=fine,compact,1,0
<br><br>
The advantage of this setting is that consecutive threads are bound close together, so that communication overhead, cache line invalidation overhead, and page thrashing are minimized. If the application also had a number of parallel regions that did not use all of the available OpenMP threads, you should avoid binding multiple threads to the same core, leaving other cores not utilized.
<br><br>
For a more detailed description of KMP_AFFINITY, please refer to [Intel® C++ developer guide](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/optimization-and-programming/openmp-support/openmp-library-support/thread-affinity-interface.html).
</details>

<details>
  <summary>KMP_BLOCKTIME</summary>
  <br>
   Recommended settings for CNN→ KMP_BLOCKTIME=0
  <br><br>
  Recommended settings for non-CNN→ KMP_BLOCKTIME=1 (user should verify empirically)
  <br><br>
  usage (shell)
  <pre>export KMP_BLOCKTIME=0 (or 1)</pre>
  tf_cnn_benchmarks usage (shell)
  <pre>OMP_NUM_THREADS=&lt;number of physical cores per socket&gt; python tf_cnn_benchmarks.py --num_intra_threads=&lt;number of physical cores per socket&gt;  --num_inter_threads=&lt;number of sockets&gt; --data_format=NCHW --kmp_affinity=granularity=fine,compact,1,0 --kmp_blocktime=0( or 1)</pre>
  This environment variable sets the time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleeping. The default value is 200ms.
<br><br>
After completing the execution of a parallel region, threads wait for new parallel work to become available. After a certain time has elapsed, they stop waiting, and sleep. Sleeping allows the threads to be used, until more parallel work becomes available, by non-OpenMP threaded code that may execute between parallel regions, or by other applications. A small <b>KMP_BLOCKTIME</b> value may offer better overall performance if application contains non-OpenMP threaded code that executes between parallel regions. A larger <b>KMP_BLOCKTIME</b> value may be more appropriate if threads are to be reserved solely for use for OpenMP execution, but may penalize other concurrently-running OpenMP or threaded applications. It is suggested to be set to 0 for convolutional neural network (CNN) based models.
</details>

<details>
  <summary>KMP_SETTINGS</summary>
  <br>
  Usage (shell)
  <pre>export KMP_SETTINGS=TRUE</pre>
  This environment variable enables (TRUE) or disables (FALSE) the printing of OpenMP run-time library environment variables during program execution.
</details>

## Enable Mixed Precision

Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training and inference to make it run faster and use less memory.
<br><br>
There are two options to enable BF16 mixed precision in TensorFlow.
<br>
1. Keras mixed precision API
2. AutoMixedPrecision oneDNN BFloat16 grappler pass through low level session configuration

Refer to <a href="https://www.intel.com/content/www/us/en/developer/articles/guide/getting-started-with-automixedprecisionmkl.html">Getting Started with Mixed Precision Support in oneDNN Bfloat16</a> for more details.

## Additional Information

<details>
  <summary>TensorFlow Operations accelerated by oneDNN</summary>
  <br>
<table><tbody><tr><td>AddN</td></tr><tr><td>AvgPool</td></tr><tr><td>AvgPool3D</td></tr><tr><td>AvgPool3DGrad</td></tr><tr><td>AvgPoolGrad</td></tr><tr><td>Conv2D</td></tr><tr><td>Conv2DBackpropFilter</td></tr><tr><td>Conv2DBackpropFilterWithBias</td></tr><tr><td>Conv2DBackpropInput</td></tr><tr><td>Conv2DWithBias</td></tr><tr><td>Conv2DWithBiasBackpropBias</td></tr><tr><td>Conv3D</td></tr><tr><td>Conv3DBackpropFilter</td></tr><tr><td>Conv3DBackpropInput</td></tr><tr><td>DepthwiseConv2dNative</td></tr><tr><td>DepthwiseConv2dNativeBackpropFilter</td></tr><tr><td>DepthwiseConv2dNativeBackpropInput</td></tr><tr><td>Dequantize</td></tr><tr><td>Einsum</td></tr><tr><td>Elu</td></tr><tr><td>EluGrad</td></tr><tr><td>FusedBatchNorm</td></tr><tr><td>FusedBatchNorm</td></tr><tr><td>FusedBatchNormFusion</td></tr><tr><td>FusedBatchNormGrad</td></tr><tr><td>FusedBatchNormGrad</td></tr><tr><td>FusedConv2D</td></tr><tr><td>FusedDepthwiseConv2dNative</td></tr><tr><td>FusedMatMul</td></tr><tr><td>LeakyRelu</td></tr><tr><td>LeakyReluGrad</td></tr><tr><td>LRN</td></tr><tr><td>LRNGrad</td></tr><tr><td>MatMul</td></tr><tr><td>MaxPool</td></tr><tr><td>MaxPool3D</td></tr><tr><td>MaxPool3DGrad</td></tr><tr><td>MaxPoolGrad</td></tr><tr><td>Mul</td></tr><tr><td>Quantize</td></tr><tr><td>QuantizedAvgPool</td></tr><tr><td>QuantizedConcat</td></tr><tr><td>QuantizedConv2D</td></tr><tr><td>QuantizedDepthwiseConv2D</td></tr><tr><td>QuantizedMatMul</td></tr><tr><td>QuantizedMaxPool</td></tr><tr><td>Relu</td></tr><tr><td>Relu6</td></tr><tr><td>Relu6Grad</td></tr><tr><td>ReluGrad</td></tr><tr><td>Softmax</td></tr><tr><td>Tanh</td></tr><tr><td>TanhGrad</td></tr></tbody></table> 
</details>

## Known issues

1. Performance degradation may be observed running with B16 on small batch size.

## Resources

Check out these resource links for more information about Intel’s AI Kit and TensorFlow optimizations:
<li><a href="https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html">Intel® oneAPI AI Analytics ToolKit (AI Kit) overview</a></li>
<li><a href="https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit-download.html?operatingsystem=Linux">AI Kit Linux* Downloads</a> and <a href="https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html">Get Started Guide</a></li>
<li><a href="https://www.intel.com/content/www/us/en/developer/tools/frameworks/overview.html#tensor-flow">Intel® Optimization for TensorFlow Framework</a> and <a href="https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html">Installation Guide</a>
