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
  <b>Recommended settings → data_format = NHWC</b>
<br>
tf_cnn_benchmarks usage (shell)
<br>
<pre>python tf_cnn_benchmarks.py --num_intra_threads=&lt;number of physical cores per socket&gt; --num_inter_threads=&lt;number of sockets&gt; --data_format=NHWC</pre>
<br>
Efficiently using cache and memory yields remarkable improvements in overall performance. A good memory access pattern minimizes extra cost for accessing data in memory and improves overall processing. Data layout, how data is stored and accessed, plays an important role in achieving these good memory access patterns. Data layout describes how multidimensional arrays are stored linearly in memory address space.

In most cases, data layout is represented by four letters for a two-dimensional image:

N: Batch size, indicates number of images in a batch.
C: Channel, indicates number of channels in an image.
W: Width, indicates number of horizontal pixels of an image.
H: Height, indicates number of vertical pixels of an image.
The order of these four letters indicates how pixel data are stored in the one-dimensional memory space. For instance, NCHW indicates pixel data are stored as width first, then height, then channel, and finally batch (Illustrated in Figure 2). The data is then accessed from left-to-right with channel-first indexing. NCHW is the recommended data layout for using oneDNN, since this format is an efficient data layout for the CPU. TensorFlow uses NHWC as its default data layout, but it also supports NCHW.

 

Data Formats for Deep Learning NHWC and NCHW

Figure 1: Data Formats for Deep Learning NHWC and NCHW

NOTE : Intel Optimized TensorFlow supports both plain data formats like NCHW/NHWC and also oneDNN blocked data format since version 2.4. Using blocked format might help on vectorization but might introduce some data reordering operations in TensorFlow.

Users could enable/disable usage of oneDNN blocked data format in Tensorflow by TF_ENABLE_MKL_NATIVE_FORMAT environment variable. By exporting TF_ENABLE_MKL_NATIVE_FORMAT=0, TensorFlow will use oneDNN blocked data format instead. Please check oneDNN memory format for more information about oneDNN blocked data format.

We recommend users to enable NATIVE_FORMAT by below command to achieve good out-of-box performance.
export TF_ENABLE_MKL_NATIVE_FORMAT=1 (or 0)
</details>

<details>
<summary>oneDNN Related Runtime Environment Variables</summary>
</details>

<details>
<summary>Memory Allocator</summary>
</details>



















<br><br>
***

## General text styling
```markdown
*This text will be italic*
_This will also be italic_

**This text will be bold**
__This will also be bold__

_You **can** combine them_
```

*This text will be italic*
_This will also be italic_

**This text will be bold**
__This will also be bold__

_You **can** combine them_

Superscript and Subscript doesn't seem to work at the moment.
```markdown
Superscript
H~2~O

Subscript
X^2^
```

Superscript example: H~2~O

Subscript example: X^2^

:bulb: 🖥️ :d

<br><br>
***
### Links
```markdown
[Intel](https://www.intel.com).
```
Example:
My favorite company is [Intel](https://www.intel.com).

mailto:
(note for spam reasons, we discourage use of email links)
[example@gitlab.com](mailto:example@gitlab.com)


<br><br>
***
### Block quotes
This doesn't work with our template. We have requested some code updates so that block quotes actually render properly on articles. I will update this file once that is done.

```markdown
> We're living the future so
> the present is our past.
```

> We're living the future so
> the present is our past.
> 
<br><br>
***
### CSS Attribution - Requested

Another Item we are exploring is the ability to call the css that is used on the site. This isn't working yet. I will update once we have the functionality in place. There are two options IT is exploring. 
We have requested that IT add this feature so we can call certain css properties into markdown.

{: .greyHighlight}
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.


<br><br>
***
### Footnotes 
Footnotes aren't working as expected. I have raised a ticket with IT to see if they can enable this feature.
```markdown
Here's a sentence with a footnote. [^1]  
  
[^1]: This is the footnote.
```
Here's a sentence with a footnote. [^2]  
  
[^2]: This is another footnote to go with the first.

<br><br>
***
## Examples of math in .md

This expression $\sum_{i=1}^n X_i$ is inlined but doesn't work at the moment.

When this is a full expression, it works fine.
$$
\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.
$$


<br><br>
***
## Code on your page
Adding code into your sentence is simple. 
```markdown
`this is your code`
```

Example
Some `inline code` if you need to put inside a sentence.


If you have javascript:
```javascript
// An highlighted block
var foo = 'bar';
```

A very common one on the DevZone is bash
```bash
export I_MPI_ROOT=/opt/intel/oneapi/lib/intel64
export PATH=${I_MPI_ROOT}/libfabric/bin:${I_MPI_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${I_MPI_ROOT}/libfabric:${I_MPI_ROOT}:$LD_LIBRARY_PATH
export FI_PROVIDER_PATH=${I_MPI_ROOT}/libfabric
```

<details>
  <summary>Expand to see the full list of available skins</summary>
  <br>

* plaintext
* abap
* actionscript
* apacheconf
* applescript
* aspnet
* bash
* basic
* c
* coffeescript
* cpp
* csharp
* css
* d
* dart
* diff
* docker
* erlang
* fortran
* fsharp
* git
* go
* groovy
* haskell
* html
* http
* ini
* java
* javascript
* lua
* makefile
* markdown
* matlab
* nginx
* objectivec
* pascal
* perl
* php
* prolog
* python
* puppet
* r
* ruby
* rust
* sas
* scala
* scheme
* sql
* swift
* twig
* vim
* xmlxhtml
* yaml
</details>


<br><br>
***
## Lists
### Creating an ordered list
1. First item  
2. Second item  
3. Third item  
4. Fourth item

This will also do the same thing
1. First item  
1. Second item  
1. Third item  
1. Fourth item

This will also do the same thing
1. First item  
8. Second item  
3. Third item  
5. Fourth item

Most of the time a MD editor will try to fix your list numbering
You can also indent by adding a few spaces.


An example of a horizontal rule
```markdown
***
```
***


<br><br>
***
### Creating Unordered Lists
- First item  
- Second item  
- Third item  
- Fourth item

**Split Lists**
- list one - item 1
- list one - item 2
     - sub item 1
     - sub item 2
- list one - item 3
<br><br>
- list two - item A
- list two - item B


<br><br>
***
## Tables

```markdown
| Syntax | Description |  
| ----------- | ----------- |  
| Header | Title |  
| Paragraph | Text |
```

| Syntax | Description |  
| ----------- | ----------- |  
| Header | Title |  
| Paragraph | Text |


<br><br>
***
## A collapsible section with markdown
This does work within the article template. THe arrow is a bit large, but I will see if there is a way to get it updated.
Code:
```markdown
<details>
  <summary>Click to expand!</summary>
  <br>
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>
```
Example
<details>
  <summary>Click to expand!</summary>
  <br>
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>


<br><br>
***

## Images

To include an image from intel.com, you can do a relative link. Make sure you put all stock imagery, marketing imagery, logos, and photos of people in intel.com where we monitor licensing and expiry dates. You can use the relative path to your image.

```markdown
![This is your Alt Text](/content/dam/www/central-libraries/us/en/images/oneapi-kits20211-4x3-rwd.png)
```

![This is your Alt Text](/content/dam/www/central-libraries/us/en/images/oneapi-kits20211-4x3-rwd.png)

<br>
You can also choose to host your screenshots, diagrams, terminal window images in your repo. Just remember, you are now supporting the live site. Don't move or delete images without updating your article. Also, make sure to use the full github URL and not a relative path.

![This is your Alt Text](https://raw.githubusercontent.com/tracyjohnsonidz/devzone-articles/main/diagram-full-workflow-16x9.webp)

<br><br>
code graphis are not available for IDZ articles.
```plantuml
!define ICONURL https://raw.githubusercontent.com/tupadr3/plantuml-icon-font-sprites/v2.1.0
skinparam defaultTextAlignment center
!include ICONURL/common.puml
!include ICONURL/font-awesome-5/gitlab.puml
!include ICONURL/font-awesome-5/java.puml
!include ICONURL/font-awesome-5/rocket.puml
!include ICONURL/font-awesome/newspaper_o.puml
FA_NEWSPAPER_O(news,good news!,node) #White {
FA5_GITLAB(gitlab,GitLab.com,node) #White
FA5_JAVA(java,PlantUML,node) #White
FA5_ROCKET(rocket,Integrated,node) #White
}
gitlab ..> java
java ..> rocket
```
<br><br>
***
## Videos
Go to the youtube video and copy the embed code. Just replace the iframe src url with your youtube video URL.

```markdown
<div>
  <div style="position:relative;padding-top:56.25%;">
    <iframe src="https://www.youtube.com/embed/c7st0drv54U" frameborder="0" allowfullscreen
      style="position:absolute;top:0;left:0;width:100%;height:100%;"></iframe>
  </div>
</div>
```

<div>
  <div style="position:relative;padding-top:56.25%;">
    <iframe src="https://www.youtube.com/embed/c7st0drv54U" frameborder="0" allowfullscreen
      style="position:absolute;top:0;left:0;width:100%;height:100%;"></iframe>
  </div>
</div>

<br>

You can also use this embed code for brightcove videos. Just replace the videoid= # in the embed code below
To find the video ID, simply right click on the video on developer.intel.com and select **Player Information**. Video ID value is listed under Source.

```markdown
<div style="position: relative; display: block; max-width: 900px;">
    <div style="padding-top: 56.25%;">
      <iframe src="https://players.brightcove.net/740838651001/default_default/index.html?videoId=6286027295001" allowfullscreen="" allow="encrypted-media" style="position: absolute; top: 0px; right: 0px; bottom: 0px; left: 0px; width: 100%; height: 100%;"></iframe>
  </div>
</div>
```

<div style="position: relative; display: block; max-width: 900px;">
  <div style="padding-top: 56.25%;">
    <iframe src="https://players.brightcove.net/740838651001/default_default/index.html?videoId=6286027295001" allowfullscreen="" allow="encrypted-media" style="position: absolute; top: 0px; right: 0px; bottom: 0px; left: 0px; width: 100%; height: 100%;"></iframe>
  </div>
</div>

multiple videos being added
<!-- blank line -->
<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/0B6m34D8cFdpMZndKTlBRU0tmczg/preview" frameborder="0" allowfullscreen="true"> </iframe>
</figure>

<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/0B6m34D8cFdpMZndKTlBRU0tmczg/preview" frameborder="0" allowfullscreen="true"> </iframe>
</figure>

<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/0B6m34D8cFdpMZndKTlBRU0tmczg/preview" frameborder="0" allowfullscreen="true"> </iframe>
</figure>
<!-- blank line -->

this is an anchor
{: #hello-world}
```markdown
{: #hello-world}

```
**Note:** a note is something that needs to be mentioned but is apart from the context.
{: .note}

Can we embed code from github, here is a gitlab test.
<!-- leave a blank line here -->
<script src="https://gitlab.com/gitlab-org/gitlab-ce/snippets/1717978.js"></script>
<!-- leave a blank line here -->

{::options parse_block_html="false" /}

<div class="center">

<blockquote class="twitter-tweet" data-partner="tweetdeck"><p lang="en" dir="ltr">Thanks to <a href="https://twitter.com/gitlab">@gitlab</a> for joining <a href="https://twitter.com/RailsGirlsCluj">@RailsGirlsCluj</a>! <a href="https://t.co/NOoiqDWKVY">pic.twitter.com/NOoiqDWKVY</a></p>&mdash; RailsGirlsCluj (@RailsGirlsCluj) <a href="https://twitter.com/RailsGirlsCluj/status/784847271645028352">October 8, 2016</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

</div>
