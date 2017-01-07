#include <iostream>
#include <fstream>
#include <iterator>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/algorithm/string.hpp>

#include <render_views_tesselated_sphere_mod.h>
#include <patch_generator.h>
#include <train_patch_generator.h>


using namespace std;

DEFINE_bool(render, false, "Render Viewpoints (input=PLY file, output=Output Folder");
DEFINE_bool(genpatches, false, "Generate Patches, input=Render Output Folders of each object, output=Output folder");
DEFINE_bool(gentrainpatches, false, "Generate Training Patches after training a network. input=folder containing patches. output=forest-patches-filename");
DEFINE_bool(lmdb, false, "Save patches in lmdb database appropriate for caffe input");
DEFINE_bool(binfile, false, "Save patches in binary format");
DEFINE_string(input, "", "Input argument");
DEFINE_string(output, "", "Output argument");
DEFINE_int32(tessel_level, 1, "Tessellation Level");
DEFINE_int32(inPlaceCamRot, 24, "In Place Camera Rotations");
DEFINE_int32(lightings, 3, "Number of times the light will increase (max 10)");
DEFINE_int32(numHeights, 4, "Number of different heights of the camera. Starting point is 0.6cm");
DEFINE_double(heightStep, 0.25, "Step of heights of the camera in m.");
DEFINE_double(startHeight, 0.3, "Starting distance of the camera in m from the center of the object.");
DEFINE_int32(patch_size, 20, "Pixel size in voxels");
DEFINE_double(voxel_size, 0.001, "Voxel size in meters");
DEFINE_int32(stride, 10, "Stride for extracting patches");
DEFINE_string(caffe_definition, "", "Caffe definition model filename");
DEFINE_string(caffe_weights, "", "Caffe weights model filename");
DEFINE_int32(gpu, -1, "Gpu device number (0). if not set, default is CPU - used only in --gentrainpatches");
DEFINE_bool(no_random_values, false, "Specify when depth=0 if a random value will be generated");
DEFINE_double(distance_threshold, 3.0f, "Do not consider patches being after this threshold away from the camera. In (m)");
DEFINE_int32(batch_size, 1, "Batch size for creating train patches");

DEFINE_bool(above_z, false, "set if camera should go only above Z axis");
DEFINE_bool(below_z, false, "set if camera should go only below Z axis");
// always set to false
DEFINE_bool(use_surface_normals, false, "Wheather extract surface normals patches of simple rgbd");
DEFINE_double(max_depth_range_in_m, 0.25f, "Max distance to truncate depth in patches");

DEFINE_double(percent, 1, "Keep percent of patches");

// these are put in order to be able
// to render a table below an object
DEFINE_bool(render_around_0, false, "Render around center of axes and not around center of object");
DEFINE_double(object_radius, -1.0, "Define another radius rather than the one defined by object extend (helps if a table is below)");


int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    #ifndef GFLAGS_GFLAGS_H_
      namespace gflags = google;
    #endif

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_render){

        ifstream f(FLAGS_input.c_str());
        CHECK(f) << "Input file " << FLAGS_input.c_str() << " not found";
        RenderViewsTesselatedSphere view_renderer;
        view_renderer.setPlyFileName(FLAGS_input);
        view_renderer.setOutFolder(FLAGS_output);
        CHECK_GT(FLAGS_tessel_level, 0);
        view_renderer.setTesselationLevel(FLAGS_tessel_level);
        CHECK_GT(FLAGS_inPlaceCamRot, 0);
        view_renderer.setInPlaceCamRotations(FLAGS_inPlaceCamRot);
        view_renderer.setLightings(FLAGS_lightings);
        view_renderer.setHeight(FLAGS_numHeights, (float)FLAGS_heightStep);
        view_renderer.setStartHeight(FLAGS_startHeight);
        view_renderer.setAboveZ(FLAGS_above_z);
        view_renderer.setBelowZ(FLAGS_below_z);
        view_renderer.setRenderAround0(FLAGS_render_around_0);
        view_renderer.setObjectRadius(FLAGS_object_radius);
        view_renderer.generateViews();

    }
    else if(FLAGS_genpatches){

        patch_generator pgen;
        std::vector<std::string> input_objects_folders;
        boost::split(input_objects_folders, FLAGS_input, boost::is_any_of(", "));        
        if(!FLAGS_lmdb && !FLAGS_binfile){
            std::cout << "No output method specified, using lmdb by default" << std::endl;
            FLAGS_lmdb = true;
        }
        if(FLAGS_lmdb){
            pgen.setOutputType(patch_generator::OUTPUT_LMDB);
        } else {
            pgen.setOutputType(patch_generator::OUTPUT_BIN);
        }
        pgen.setInputObjectFolders(input_objects_folders);
        pgen.setOutputFolder(FLAGS_output);
        pgen.setVoxelSizeInM(FLAGS_voxel_size);
        pgen.setPatchSizeInVoxels(FLAGS_patch_size);
        pgen.setStride(FLAGS_stride);
        pgen.setGenerateRandomValues(!FLAGS_no_random_values);
        pgen.setDistanceThreshold(FLAGS_distance_threshold);
        pgen.setPercent(FLAGS_percent);
        if(FLAGS_use_surface_normals)
            pgen.generatePatches();
        else {
            pgen.setMaxDepthRangeInM(FLAGS_max_depth_range_in_m);
            pgen.generatePatches_rgbd();
        }

    }

    else if(FLAGS_gentrainpatches){

        CHECK_GT(FLAGS_caffe_definition.size(), 0) << "No caffe definition model defined.";
        CHECK_GT(FLAGS_caffe_weights.size(), 0) << "No caffe weights model defined.";
        CHECK_GT(FLAGS_input.size(), 0) << "No input lmdb specified.";
        CHECK_GT(FLAGS_output.size(), 0) << "No output file specified.";
        train_patch_generator trainpg;
        trainpg.setCaffeModel(FLAGS_caffe_definition, FLAGS_caffe_weights);

        if(FLAGS_gpu >= 0)
            trainpg.useGPU(FLAGS_gpu);
        else
            trainpg.useCPU();

        trainpg.setInputLmdb(FLAGS_input);
        trainpg.setOutputFile(FLAGS_output);
        trainpg.setPatchSize(FLAGS_patch_size);
        trainpg.setBatchSize(FLAGS_batch_size);
        trainpg.generate_train_patches();        
    }

    return 0;
}

