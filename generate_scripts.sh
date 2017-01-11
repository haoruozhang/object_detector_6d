#!/bin/sh

##################
#### BINARIES ####
##################
# define paths for the binaries
OBJ_DETECTOR_PATH=$(pwd)/build
CAFFE_BINARY=$HOME/apps/caffe-master/build/tools/caffe

# define the folder with the meshes (.ply)
MESH_FOLDER=meshes

# Set default options for all objects
# If you need a specific option for some
# objects only, edit the generated script files

###################
#### RENDERING ####
###################
# first distance from camera. 0.6 is the closest
# you can get with Xtion
RENDER_START_HEIGHT=0.6
# maximum number of different distance from the camera
# to render the object
RENDER_NUM_HEIGHTS=2
# difference between each distance
RENDER_HEIGHT_STEP=0.2
# simulate brightness, in each lighting the image
# gets brighter
RENDER_LIGHTINGS=1
# number of camera rotations in each position on the sphere
RENDER_IN_PLACE_CAM_ROT=18
# We assume the objects are standing along the Z axis in their
# natural position. Set to true only if you expect the objects 
# to be sitting upfront. You can change it later in render.sh
# if you want a different setting for each object.
RENDER_ABOVE_Z=false

##########################
#### PATCH EXTRACTION ####
##########################
# the patch is quantized into PATCH_SIZExPATCH_SIZE cells
PATCH_SIZE=8
# each cell is VOXEL_SIZE meters. 
# the actual length of the side of the patch is PATCH_SIZE*VOXEL_SIZE
# be careful not to exceed object dimensions
VOXEL_SIZE=0.005
# max distance from the center of patch to include in the patch
# usually set to the max radius of biggest object
MAX_DEPTH_RANGE=0.25
# stride in pixels
STRIDE_IN_TRAIN=2
STRIDE_IN_TEST=2
# Set to true if your objects are segmented from the background.
# Otherwise random values should be generated at the borders 
# of the objects.
ARE_OBJECTS_SEGMENTED=false
# percentage of all patches to use to train the autoencoder
CAFFE_PERCENT=0.1
# GPU Device number. Set to -1 to use CPU
GPU_DEVICE=0

################
#### CAFFE #####
################
# Should match with the caffe_solver.prototxt
ITERATIONS=100000
# Batch size
BATCH_SIZE=100
# for more options edit the caffe_solver.prototxt

######################
#### HOUGH FOREST ####
######################
# number of threads to train one tree
THREADS_PER_TREE=8
# number of trees to process in parallel
# Total number of threads used are THREADS_PER_TREE*PARALLEL_TREES.
# If you have at most 8 threads, it is better to use PARALLEL_TREES=1
# and set THREADS_PER_TREE=max.number of cores.
# If you have more, i.e. 32, you can use THREADS_PER_TREE=8 and
# PARALLEL_TREES=4
PARALLEL_TREES=1
# Number of threads in testing
THREADS_IN_TEST=8
# Number of trees to train (use a multiple of PARALLEL_TREES)
NTREES=4


################################
##### MORE TESTING OPTIONS #####
################################
# Camera Intrinsics
FX=575
FY=575
CX=319.5
CY=239.5
# For more options, open the generated proto file
# in the test folder. For details about the options
# see the documentation of GitHub or the definition
# proto file in HoughForest/include/proto

files="$(ls -A $MESH_FOLDER/*.ply)" 
if [ -z "$files" ]; then
  echo "No .ply files found in $MESH_FOLDER"
  exit 1
fi

cur_folder="$(pwd)"

if [ ! -d "training" ]; then
  TRAIN_FOLDER=training
  TEST_FOLDER=test
else
  i=2
  while [ -d "training$i" ]; do
    i=$(expr $i + 1)
  done
  TRAIN_FOLDER="training$i"
  TEST_FOLDER="test$i"
fi

mkdir -p $TRAIN_FOLDER
mkdir -p $TEST_FOLDER
rm -rf $TEST_FOLDER/*

cd $TRAIN_FOLDER

mkdir -p "renderings"

obj_str=""
echo "Generating Scripts..."
echo "Objects found:"
for f in $files 
do
  obj_name="${f%.ply}"
  obj_name=${obj_name##*/}
  echo "$obj_name"
  mkdir -p "renderings/$obj_name"
  printf "# $obj_name\necho \"Rendering object: $obj_name\"\n$OBJ_DETECTOR_PATH/PatchGen --render" >> render.sh
  printf " \\" >> render.sh
  printf "\n--input=../$f" >> render.sh
  printf " \\" >> render.sh
  printf "\n--output=renderings/$obj_name" >> render.sh
  printf " \\" >> render.sh
  printf "\n--startHeight=$RENDER_START_HEIGHT" >> render.sh
  printf " \\" >> render.sh
  printf "\n--heightStep=$RENDER_HEIGHT_STEP" >> render.sh
  printf " \\" >> render.sh
  printf "\n--lightings=$RENDER_LIGHTINGS" >> render.sh
  printf " \\" >> render.sh
  printf "\n--numHeights=$RENDER_NUM_HEIGHTS" >> render.sh
  printf " \\" >> render.sh
  printf "\n--inPlaceCamRot=$RENDER_IN_PLACE_CAM_ROT" >> render.sh
  if $RENDER_ABOVE_Z; then
      printf " \\" >> render.sh
      printf "\n--above_z" >> render.sh
  fi
  printf "\n\n" >> render.sh
  if [ ! -z $obj_str ]; then
    obj_str="$obj_str,"
  fi
  obj_str="${obj_str}renderings/$obj_name"

  printf "object_options {
name: \"$obj_name\"
mesh_file: \"$cur_folder/$f\"
instances: 1
nn_search_radius:0.015
icp_iterations: 60
max_location_hypotheses: 12
should_detect: true
}\n" >> ../$TEST_FOLDER/detector_options.proto


done

caffe_patches_folder="patches_caffe_p${PATCH_SIZE}_v${VOXEL_SIZE}"
all_patches_folder="patches_full_p${PATCH_SIZE}_v${VOXEL_SIZE}"
mkdir -p $caffe_patches_folder
mkdir -p $all_patches_folder

extract_str="# generate patches for autoencoder training\n$OBJ_DETECTOR_PATH/PatchGen --genpatches --lmdb --input=$obj_str --voxel_size=$VOXEL_SIZE --patch_size=$PATCH_SIZE --max_depth_range_in_m=$MAX_DEPTH_RANGE --stride=$STRIDE_IN_TRAIN"
if ($ARE_OBJECTS_SEGMENTED); then
  extract_str="$extract_str --no_random_values"
fi

printf "$extract_str --output=$all_patches_folder" > extract_patches_full.sh
extract_str="$extract_str --percent=$CAFFE_PERCENT"
printf "$extract_str --output=$caffe_patches_folder" > extract_patches_caffe.sh


PROTO1="name: \"PATCHAutoencoder\"
layers {
  top: \"data\"
  name: \"data\"
  type: DATA
  data_param {
    source: \"$caffe_patches_folder\"
    backend: LMDB
    batch_size: $BATCH_SIZE
  }
  transform_param {
    scale: 0.0039215684
  }
  include: { phase: TRAIN }
}
layers {
  top: \"data\"
  name: \"data\"
  type: DATA
  data_param {
    source: \"./$caffe_patches_folder\"
    backend: LMDB
    batch_size: $BATCH_SIZE
  }
  transform_param {
    scale: 0.0039215684
  }
  include: {
    phase: TEST
    stage: \"test-on-train\"
  }
}

layers {
  bottom: \"data\"
  top: \"flatdata\"
  name: \"flatdata\"
  type: FLATTEN
}
layers {
  bottom: \"data\"
  top: \"encode1\"
  name: \"encode1\"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 1
  weight_decay: 1
  weight_decay: 0
  inner_product_param {
    num_output: 1500
    weight_filler {
      type: \"gaussian\"
      std: 1
      sparse: 40
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}

layers {
  bottom: \"encode1\"
  top: \"encode1neuron\"
  name: \"encode1neuron\"
  type: SIGMOID
}
layers {
  bottom: \"encode1neuron\"
  top: \"encode2\"
  name: \"encode2\"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 1
  weight_decay: 1
  weight_decay: 0
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: \"gaussian\"
      std: 1
      sparse: 40
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layers {
  bottom: \"encode2\"
  top: \"encode2neuron\"
  name: \"encode2neuron\"
  type: SIGMOID
}

layers {
  bottom: \"encode2neuron\"
  top: \"encode3\"
  name: \"encode3\"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 1
  weight_decay: 1
  weight_decay: 0
  inner_product_param {
    num_output: 800
    weight_filler {
      type: \"gaussian\"
      std: 1
      sparse: 40
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layers {
  bottom: \"encode3\"
  top: \"encode3neuron\"
  name: \"encode3neuron\"
  type: SIGMOID
}

layers {
  bottom: \"encode3neuron\"
  top: \"decode3\"
  name: \"decode3\"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 1
  weight_decay: 1
  weight_decay: 0
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: \"gaussian\"
      std: 1    
      sparse: 40  
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layers {
  bottom: \"decode3\"
  top: \"decode3neuron\"
  name: \"decode3neuron\"
  type: SIGMOID
}


layers {
  bottom: \"decode3neuron\"
  top: \"decode2\"
  name: \"decode2\"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 1
  weight_decay: 1
  weight_decay: 0
  inner_product_param {
    num_output: 1500
    weight_filler {
      type: \"gaussian\"
      std: 1    
      sparse: 40  
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layers {
  bottom: \"decode2\"
  top: \"decode2neuron\"
  name: \"decode2neuron\"
  type: SIGMOID
}

layers {
  bottom: \"decode2neuron\"
  top: \"decode1\"
  name: \"decode1\"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 1
  weight_decay: 1
  weight_decay: 0
  inner_product_param {
    num_output: 256
    weight_filler {
      type: \"gaussian\"
      std: 1      
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layers {
  bottom: \"decode1\"
  bottom: \"flatdata\"
  top: \"cross_entropy_loss\"
  name: \"loss\"
  type: SIGMOID_CROSS_ENTROPY_LOSS
  loss_weight: 1
}
layers {
  bottom: \"decode1\"
  top: \"decode1neuron\"
  name: \"decode1neuron\"
  type: SIGMOID
}
layers {
  bottom: \"decode1neuron\"
  bottom: \"flatdata\"
  top: \"l2_error\"
  name: \"loss\"
  type: EUCLIDEAN_LOSS
  loss_weight: 0
}"

echo "$PROTO1" > patch_autoencoder.prototxt

PROTO2="name: \"PATCHAutoencoder\"
input: \"data\"
input_dim: 100
input_dim: 4
input_dim: $PATCH_SIZE
input_dim: $PATCH_SIZE

layers {
  bottom: \"data\"
  top: \"flatdata\"
  name: \"flatdata\"
  type: FLATTEN
}
layers {
  bottom: \"data\"
  top: \"encode1\"
  name: \"encode1\"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 1
  weight_decay: 1
  weight_decay: 0
  inner_product_param {
    num_output: 1500
    weight_filler {
      type: \"gaussian\"
      std: 1
      sparse: 40
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}

layers {
  bottom: \"encode1\"
  top: \"encode1neuron\"
  name: \"encode1neuron\"
  type: SIGMOID
}
layers {
  bottom: \"encode1neuron\"
  top: \"encode2\"
  name: \"encode2\"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 1
  weight_decay: 1
  weight_decay: 0
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: \"gaussian\"
      std: 1
      sparse: 40
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layers {
  bottom: \"encode2\"
  top: \"encode2neuron\"
  name: \"encode2neuron\"
  type: SIGMOID
}



layers {
  bottom: \"encode2neuron\"
  top: \"encode3\"
  name: \"encode3\"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 1
  weight_decay: 1
  weight_decay: 0
  inner_product_param {
    num_output: 800
    weight_filler {
      type: \"gaussian\"
      std: 1
      sparse: 40
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layers {
  bottom: \"encode3\"
  top: \"encode3neuron\"
  name: \"encode3neuron\"
  type: SIGMOID
}"

echo "$PROTO2" > patch_autoencoder_half.prototxt

cp ../caffe_solver.prototxt .

echo "$CAFFE_BINARY train --solver=caffe_solver.prototxt" > train_caffe.sh
echo "$OBJ_DETECTOR_PATH/PatchGen --gentrainpatches --batch_size=$BATCH_SIZE --input=$all_patches_folder --output=$all_patches_folder/patches.forest --gpu=$GPU_DEVICE --caffe_definition=patch_autoencoder_half.prototxt --caffe_weights=autoencoder_iter_$ITERATIONS.caffemodel" > create_train_patches.sh
echo "mkdir -p forest
cd forest
$OBJ_DETECTOR_PATH/HoughForest --train --threads_per_tree=$THREADS_PER_TREE --threads_for_parallel_trees=$PARALLEL_TREES --input=../$all_patches_folder/patches.forest --trees=$NTREES --patch_size_in_voxels=$PATCH_SIZE --voxel_size_in_m=$VOXEL_SIZE
cd .." > train_forest.sh


caffe_weights="autoencoder_iter_${ITERATIONS}.caffemodel"

cd ../$TEST_FOLDER
printf "caffe_definition: \"$cur_folder/$TRAIN_FOLDER/patch_autoencoder_half.prototxt\"
caffe_weights: \"$cur_folder/$TRAIN_FOLDER/$caffe_weights\"
forest_folder: \"$cur_folder/$TRAIN_FOLDER/forest\"
num_threads: $THREADS_IN_TEST
stride: $STRIDE_IN_TEST
max_depth_range_in_patch_in_m: $MAX_DEPTH_RANGE
gpu: $GPU_DEVICE
batch_size: $BATCH_SIZE
fx: $FX
fy: $FY
cx: $CX
cy: $CY
search_single_object_instance: false
search_single_object_in_group: false
use_color_similarity: true
similarity_coeff: 10
inliers_coeff: 2.5
clutter_coeff: 1.4
location_score_coeff: 1.4
pose_score_coeff: 0.7
group_total_explain_coeff: 0.5
group_common_explain_coeff: 0.3
inliers_threshold: 0.6
clutter_threshold: 0.6
final_score_threshold: 10
cluster_eps_angle_threshold: 0.05
cluster_min_points: 5
cluster_curvature_threshold: 0.1
cluster_tolerance_near: 0.03
cluster_tolerance_far: 0.05
distance_threshold: 1.5
are_objects_segmented: $ARE_OBJECTS_SEGMENTED \n" >> detector_options.proto
cd ..

if [ "$1" == "run" ]; then
  echo "running scripts.."
  cd $TRAIN_FOLDER
  bash render.sh
  bash extract_patches_caffe.sh
  bash extract_patches_full.sh
  bash train_caffe.sh
  bash create_train_patches.sh
  bash train_forest.sh
  cd ..
fi

