#include <vtkCellData.h>
#include <vtkWorldPointPicker.h>
#include <vtkPropPicker.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkTriangle.h>
#include <vtkTransform.h>
#if VTK_MAJOR_VERSION==6 || (VTK_MAJOR_VERSION==5 && VTK_MINOR_VERSION>4)
#include <vtkHardwareSelector.h>
#include <vtkSelectionNode.h>
#else
#include <vtkVisibleCellSelector.h>
#endif
#include <vtkSelection.h>
#include <vtkCellArray.h>
#include <vtkTransformFilter.h>
#include <vtkCamera.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkPointPicker.h>
#include <vtkPLYReader.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include "vtkProperty.h"

#include <cv.h>
#include <highgui.h>

#include <sstream>
#include <fstream>

#include <render_views_tesselated_sphere_mod.h>

Eigen::Matrix3f RenderViewsTesselatedSphere::getRotMatAroundVector(const Eigen::Vector3f &v, double degrees){

    Eigen::Vector3f vnorm = v.normalized();
    double theta = degrees/180.0 * 3.14159265359;
    Eigen::Matrix3f rot_mat;

    double cost = cos(theta);
    double sint = sin(theta);
    double vx = vnorm[0];
    double vy = vnorm[1];
    double vz = vnorm[2];

    rot_mat(0, 0) = cost + pow(vx,2)*(1-cost);
    rot_mat(0, 1) = vx*vy*(1-cost) - vz*sint;
    rot_mat(0, 2) = vx*vz*(1-cost) + vy*sint;
    rot_mat(1, 0) = vy*vx*(1-cost) + vz*sint;
    rot_mat(1, 1) = cost + pow(vy,2)*(1-cost);
    rot_mat(1, 2) = vy*vz*(1-cost) - vx*sint;
    rot_mat(2, 0) = vz*vx*(1-cost) - vy*sint;
    rot_mat(2, 1) = vz*vy*(1-cost) + vx*sint;
    rot_mat(2, 2) = cost + pow(vz,2)*(1-cost);

    return rot_mat;

}

void RenderViewsTesselatedSphere::save_rendering(const vtkSmartPointer<vtkRenderWindow> &render_win,
                                                 const vtkSmartPointer<vtkRenderer> &renderer,
                                                 vtkSmartPointer<vtkCamera> &cam,
                                                 int fcounter)
{


    //write rgb png
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(render_win);
    windowToImageFilter->SetMagnification(1);
    windowToImageFilter->SetInputBufferTypeToRGB();

    vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
    std::stringstream fname;
    fname << outFolder_ << "/rgb" << fcounter << ".png";    
    writer->SetFileName(fname.str().c_str());
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();


    //write depth png
    render_win->GetZbufferData (0, 0, resolutionX_ - 1, resolutionY_ - 1, &(depthbuf_[0]));

    double z_near, z_far;
    cam->GetClippingRange(z_near, z_far);

    cv::Mat depthimg = cv::Mat::zeros(480, 640, CV_16U);   


    for (int x = 0; x < resolutionX_; x++)
    {
      for (int y = 0; y < resolutionY_; y++)
      {
        float value = depthbuf_[y * resolutionX_ + x];
        if (value != 1.0)
        {
            double depth = -1 * ((z_far * z_near) / (value * (z_far - z_near) - z_far));
            depthimg.at < int16_t > (y, x) = (int16_t) (depth * 1000.0);            

        }
      }
    }   

    cv::flip(depthimg, depthimg, 0);
    fname.str("");
    fname << outFolder_ << "/depth" << fcounter << ".png";
    cv::imwrite(fname.str(), depthimg);

    //write surface normals
    std::vector<unsigned short> depthvec;
    depthvec.assign((unsigned short*)depthimg.datastart, (unsigned short*)depthimg.dataend);
    std::vector<float> normals;
    surface_normals_gpu::generate_normals(depthvec, depthimg.cols, depthimg.rows, 575.0f, normals);
    //surface_normals_gpu::generate_normals(depthvec, depthimg.cols, depthimg.rows, getRenderFocalLength(), normals);

    fname.str("");
    fname << outFolder_ << "/surface_normals" << fcounter << ".bin";
    std::ofstream snout(fname.str().c_str(), std::ios::out | std::ios::binary);
    snout.write(reinterpret_cast<const char*>(&normals[0]), normals.size()*sizeof(float));
    snout.close();


    //write pose
    fname.str("");
    fname << outFolder_ << "/pose" << fcounter << ".txt";
    ofstream fpose(fname.str().c_str());
    for(int i=0; i<4; ++i){
        for(int j=0; j<4; ++j){
            fpose << cam->GetViewTransformMatrix()->GetElement(i, j);
            if(j!=3) fpose << " ";
        }
        fpose << endl;
    }
    fpose.close();
}

void RenderViewsTesselatedSphere::generateViews() {

    //read PLY
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName (plyFileName_.c_str());

    vtkSmartPointer<vtkPolyData> vtkpolydata = reader->GetOutput();
    vtkpolydata->Update();

    //calculate center of object
    double CoM[3];
    vtkIdType npts_com = 0, *ptIds_com = NULL;
    vtkSmartPointer<vtkCellArray> cells_com = vtkpolydata->GetPolys ();

    double center[3], p1_com[3], p2_com[3], p3_com[3], area_com, totalArea_com = 0;
    double comx = 0, comy = 0, comz = 0;
    for (cells_com->InitTraversal (); cells_com->GetNextCell (npts_com, ptIds_com);)
    {
        vtkpolydata->GetPoint (ptIds_com[0], p1_com);
        vtkpolydata->GetPoint (ptIds_com[1], p2_com);
        vtkpolydata->GetPoint (ptIds_com[2], p3_com);
        vtkTriangle::TriangleCenter (p1_com, p2_com, p3_com, center);
        area_com = vtkTriangle::TriangleArea (p1_com, p2_com, p3_com);
        comx += center[0] * area_com;
        comy += center[1] * area_com;
        comz += center[2] * area_com;
        totalArea_com += area_com;
    }

    if(render_around_0_) {
        CoM[0] = CoM[1] = CoM[2] = 0.0;
    }
    else {
        CoM[0] = comx / totalArea_com;
        CoM[1] = comy / totalArea_com;
        CoM[2] = comz / totalArea_com;
    }

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
    mapper->SetInputConnection (reader->GetOutputPort());
    mapper->Update ();


    //Calculate radius according to mesh bounds
    double bb[6];
    mapper->GetBounds (bb);
    double ms = (std::max) ((std::fabs) (bb[0] - bb[1]),
                          (std::max) ((std::fabs) (bb[2] - bb[3]), (std::fabs) (bb[4] - bb[5])));

    //radius_sphere_ = ms * 2.0;
    //xtion initial distance from the object
    if(object_radius_ < 0) {
        radius_sphere_ = ms + start_height_;
    }
    else {
        radius_sphere_ = object_radius_ + start_height_;
    }

    //create icosahedron
    vtkSmartPointer<vtkPlatonicSolidSource> ico = vtkSmartPointer<vtkPlatonicSolidSource>::New ();
    ico->SetSolidTypeToIcosahedron ();
    ico->Update ();

    //tesselate cells from icosahedron
    vtkSmartPointer<vtkLoopSubdivisionFilter> subdivide = vtkSmartPointer<vtkLoopSubdivisionFilter>::New ();
    subdivide->SetNumberOfSubdivisions (tesselation_level_);
    subdivide->SetInputConnection (ico->GetOutputPort ());
    #if VTK_MAJOR_VERSION>=6
    subdivide->Update();
    #endif

    // Get camera positions
    vtkPolyData *sphere = subdivide->GetOutput ();
    #if VTK_MAJOR_VERSION<6
    sphere->Update ();
    #endif  

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > cam_positions;
    if (!use_vertices_)
    {
        vtkSmartPointer<vtkCellArray> cells_sphere = sphere->GetPolys ();
        //cam_positions.resize (sphere->GetNumberOfPolys ());

        size_t i=0;
        for (cells_sphere->InitTraversal (); cells_sphere->GetNextCell (npts_com, ptIds_com);)
        {
            sphere->GetPoint (ptIds_com[0], p1_com);
            sphere->GetPoint (ptIds_com[1], p2_com);
            sphere->GetPoint (ptIds_com[2], p3_com);
            vtkTriangle::TriangleCenter (p1_com, p2_com, p3_com, center);
            //cam_positions[i] = Eigen::Vector3f (float (center[0]), float (center[1]), float (center[2]));
            if( (above_z_ && center[2] >= 0) || (below_z_ && center[2] <= 0) || (!above_z_ && !below_z_) )
                cam_positions.push_back(Eigen::Vector3f (float (center[0]), float (center[1]), float (center[2])) );
            i++;
        }

    }
    else
    {
        //cam_positions.resize (sphere->GetNumberOfPoints ());
        for (int i = 0; i < sphere->GetNumberOfPoints (); i++)
        {
            double cam_pos[3];
            sphere->GetPoint (i, cam_pos);
            //cam_positions[i] = Eigen::Vector3f (float (cam_pos[0]), float (cam_pos[1]), float (cam_pos[2]));
            if( (above_z_ && cam_pos[2] >= 0) || (below_z_ && cam_pos[2] <= 0) || (!above_z_ && !below_z_) )
            //if(!above_z_ || (above_z_ && cam_pos[2] >= 0) )
                cam_positions.push_back( Eigen::Vector3f (float (cam_pos[0]), float (cam_pos[1]), float (cam_pos[2])) );
        }
    }

    std::cout << "Total number of viewpoints: " <<
                 cam_positions.size() * inPlaceCamRotations_ * heights_ * lightings_ << std::endl;


    //create renderer and window
    vtkSmartPointer<vtkRenderWindow> render_win = vtkSmartPointer<vtkRenderWindow>::New ();
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New ();
    render_win->AddRenderer (renderer);
    render_win->SetSize (resolutionX_, resolutionY_);
    renderer->SetBackground (1.0, 1.0, 1.0);
    vtkSmartPointer<vtkActor> actor_view = vtkSmartPointer<vtkActor>::New ();

    actor_view->SetMapper (mapper);    
    renderer->AddActor (actor_view);

    //create camera
    vtkSmartPointer<vtkCamera> cam = vtkSmartPointer<vtkCamera>::New ();
    cam->SetViewAngle (view_angle_);
    cam->SetFocalPoint (CoM[0], CoM[1], CoM[2]);

    double cam_pos[3];
    int fcounter = 0;


    for (size_t i = 0; i < cam_positions.size (); i++)
    {

        for(int cur_height=0; cur_height<heights_; ++cur_height){

            cam_pos[0] = cam_positions[i][0];
            cam_pos[1] = cam_positions[i][1];
            cam_pos[2] = cam_positions[i][2];

            Eigen::Vector3f cam_pos_3f (static_cast<float> (cam_pos[0]),
                    static_cast<float> (cam_pos[1]), static_cast<float> (cam_pos[2]));
            cam_pos_3f = cam_pos_3f.normalized ();

            for (int k = 0; k < 3; k++)
            {
              cam_pos[k] = cam_pos_3f[k] * (radius_sphere_ + cur_height*height_step_);
            }

            //Get various CameraViewUp vectors in each location

            //Get first vector randomly
            Eigen::Vector3f CamViewUp;
            if(std::fabs(cam_pos[2]) > 0.00001){
                CamViewUp << 1, 1, (- cam_pos[0] - cam_pos[1]) / cam_pos[2];
            }
            else if(std::fabs(cam_pos[1]) > 0.00001){
                CamViewUp << 1, (-cam_pos[0] - cam_pos[2]) / cam_pos[1] ,1;
            }
            else{
                CamViewUp << 0, 1, 0;
            }
            CamViewUp = CamViewUp.normalized();

            Eigen::Matrix3f rotMat = getRotMatAroundVector(cam_pos_3f, 360.0/(double)inPlaceCamRotations_);


            for(int k=0; k<3; ++k)
                cam_pos[k] += CoM[k];

            cam->SetPosition (cam_pos);
            //cam->Modified ();

            for(int r = 0; r < inPlaceCamRotations_; ++r){

                cam->SetViewUp (CamViewUp[0], CamViewUp[1], CamViewUp[2]);
                cam->Modified ();

                renderer->SetActiveCamera (cam);

                for(int light = 0; light < lightings_; ++light){

                    actor_view->GetProperty()->SetAmbient(light * 0.1);

                    render_win->Render ();

                    save_rendering(render_win, renderer, cam, fcounter++);
                    std::cout << "Rendering viewpoint: " << fcounter << "\r";
                }

                CamViewUp = rotMat * CamViewUp;

            }

        }

    }

}
