# 6D Object Detector

The object detector is able to recognise objects in 3D space as well as their pose, from depth camera input. It is based on the following paper:

[**Recovering 6D Object Pose and Predicting Next-Best-View in the Crowd**](http://www.iis.ee.ic.ac.uk/rkouskou/research/6D_NBV.html)
*Andreas Doumanoglou, Rigas Kouskouridas, Sotiris Malassiotis, Tae-Kyun Kim
CVPR 2016*

but it has been modified for the needs of various projects. Therefore there might be differences from the paper, and it is not guaranteed that one can reproduce the results provided in the paper exactly. Unfortunately, the values of all the parameters that were used to run the experiments of the paper have been overwritten, but the default values should be close to them. However, one should search for the optimal parameter values that work best for the objects of interest.
If you use this source code to evaluate the method on your own test scenario, please cite the above paper.

Please read the guidelines carefully in order to use the detector properly.

## Building the project

The source code is tested on Ubuntu 14.04. Below are all the required dependencies:
* Glog
* GFlags
* OpenMP
* Boost
* OpenCV (2.4.10)
* PCL
* VTK (5.10)
* CUDA
* LMDB
* Protobuf
* Caffe (1.7)

When you have installed all the required libraries, build the project running the following commands:
```bash
mkdir build
cd build
cmake ..
make
```
If no error was generated, two binaries should have been created: **PatchGen** and **HoughForest**. 

In case you get an error not finding the headers of caffe, try the solution [here](https://github.com/BVLC/caffe/issues/1761).

## Training

The detector requires CAD models of the objects in order to recognise them in real images. The models must be provided in PLY format. For simplicity of the code, the ply files must have the following characteristics:

- Must be in plain text (not binary).
- Must contain only point coordinates (X,Y,Z) and colour per vertex (R,G,B). No normals and no texture.
- In case you have the color in texture image files, meshlab is able to convert texture to color per vertex.
- In case you have more than one texture images, the version of meshlab I was using (1.3.3) has a bug during the conversion, therefore I have included a script in *matlab_ply_merge_textures/* to concatenate all texture images to one. More details are written at the top of the script.
- If the model contains few points with large triangles, it is recommended to populate your model with more points since there are point-to-point validation checks and the detector does not produce more points on its own. Blender can be also used for this purpose.
- If your models contain holes at the bottom of the objects, there are options in the “generate_scripts.sh” file to render above or below the z axis of the object to avoid rendering the views with the hole.

You can find examples of proper ply models in *meshes/* folder.

Put your ply files in the folder *meshes/* . The filename is used as the name of the object, but you can also change it later. Then edit the generate_scripts.sh file and modify the options at the top to meet your needs. A short description of each option is included in the file. Then run 
```bash
bash generate_scripts.sh 
```
to generate all the required scripts for training, or 
```bash
bash generate_scripts.sh run
```
to run the scripts after they are generated. This script generates two folders called *trainingXX* and *testingXX*. After the training scripts run successfully, the neural network along with the forest are stored in the *trainingXX* folder, and the *testingXX* folder will contain a file called *object_options.proto* with all the options needed for the binary HoughForest to execute and recognize the objects. You can still modify this file if you want to change the options again.  

The final stage includes a Hough Forest training. The implementation of training each tree has been optimized for parallel processing. The parallelization is happening across all the samples of all the nodes of each depth of the tree, which makes it efficient in all stages of the training. This is because it is independent from the number of nodes per depth, or the number of samples per node, which is the common way to parallelize training of a random tree.
Unfortunately it is not a separate library, but one can use just this part of the code, which is somewhat generic, for his own purposes.

## Testing

To detect your trained objects on a scene, you first need to start the HoughForest in testing mode and provide the options file that was generated during training, for example:

```bash
./HoughForest --test --object_options=testing/object_options.proto
```

The detector loads the forest and the network, and when finished it is waiting for the input images. You should provide two strings, the first containing the path to the RGB image, and the second the path to the corresponding depth image. After you enter these two strings (with an enter between them) the detection starts. By default, the output is written to the folder you execute the binary, but you can specify your output directory by using the option *--output_dir=/your/output/folder*. In case you want to test multiple images, you should construct a file containing the RGB and depth images one under the other, and execute the binary with the option *< your_input_images.txt*.

## Setting Parameters

It is highly recommended that before starting your evaluation on all your test images, to debug and manually try to find some good parameters for your test scene and objects using one or two test images.

The first set of parameters is about the scene clustering. The scene is segmented into smooth clusters, and what you should try to achieve by this segmentation is the following:
Smooth regions such as flat surfaces should form individual clusters. Examples are the floor, walls, the surface of a table and smooth parts of the objects. If a smooth surface if segmented into more than one cluster is fine, but if for example a cluster contains both the table and your object, you should change your parameters in order to separate them into different clusters.

The second set of parameters contains the coefficients of the cost function. The general form of this function is:
*final_score = similarity_score * similarity_coeff + inliers_ratio * inliers_coeff + clutter_score * clutter_coeff + pose_score * pose_score_coeff + location_score * location_score_coeff*
For the scores, refer to the paper. You can modify all the coefficients inside the the objects_options.proto file generated from the *generate_scripts.sh* script. The default values are a good place to start. The way that these values should be set is to debug few hypotheses generated using the option (--visualize_hypotheses) and try to find the relative importance of each term, comparing the final score of good and bad hypotheses on the same object. That is, if the parameters are set properly, the good hypotheses should have a better final score than the bad ones.

The third set of parameters are the ones used for training. These include:

- Number of trees in the forest. The more the better, but after 8 trees usually there is not significant improvement, while the detector gets slower.
- Patch Size. The patch size shouldn't exceed the 2/3 of the largest object diameter if you want the detector to perform well in cases of occlusion.
- Render height. It is recommended that you render the models in a distance from the camera that actually appear in your test images. However the algorithm is scale invariant and can handle cases where the training scale does not match the testing scale.

Some more exaplanations can be found in *HoughForest/include/proto/detector_options.proto* where all the options are defined.




