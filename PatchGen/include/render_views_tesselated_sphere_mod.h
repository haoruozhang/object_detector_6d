/*
 * render_views_tesselated_sphere.h
 *
 *  Created on: Dec 23, 2011
 *      Author: aitor
 *
 * Slightly modified by Andreas Doumanoglou
 */

#ifndef RENDER_VIEWS_TESSELATED_SPHERE_H_
#define RENDER_VIEWS_TESSELATED_SPHERE_H_

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

#include <Eigen/Dense>

#include <cuda/surface_normals.h>


class RenderViewsTesselatedSphere
{
private:  
  std::string plyFileName_;
  std::string outFolder_;
  int resolutionX_, resolutionY_;
  int tesselation_level_;
  bool use_vertices_;
  float view_angle_;
  float radius_sphere_;
  int inPlaceCamRotations_;  
  int lightings_;
  int heights_;
  float height_step_;
  float start_height_;
  bool above_z_;
  bool below_z_;
  std::vector<float> depthbuf_;

  bool render_around_0_;
  double object_radius_;

  Eigen::Matrix3f getRotMatAroundVector(const Eigen::Vector3f& v, double degrees);

  struct camPosConstraintsAllTrue
  {
    bool
    operator() (const Eigen::Vector3f & /*pos*/) const
    {
      return true;
    }

  };

  void save_rendering(const vtkSmartPointer<vtkRenderWindow> &render_win, const vtkSmartPointer<vtkRenderer> &renderer, vtkSmartPointer<vtkCamera> &cam, int fcounter);

  float getRenderFocalLength(){
      return (float)resolutionY_ / 2.0f / (float)tan(view_angle_/ 180.0f * 3.141592f / 2.0f);
  }

public:
  RenderViewsTesselatedSphere ()
  {
    resolutionX_ = 640;
    resolutionY_ = 480;
    tesselation_level_ = 2;
    use_vertices_ = true;
    view_angle_ = 45.3105;
    inPlaceCamRotations_ = 24;    
    start_height_ = 0.3;
    depthbuf_.resize(resolutionX_ * resolutionY_);
    above_z_ = false;
    below_z_ = false;
    render_around_0_ = false;
    object_radius_ = -1.0;
  }

  void setPlyFileName(const std::string& filename){
      plyFileName_ = filename;
  }

  void setOutFolder(const std::string &folder){
      outFolder_ = folder;
  }

    void setInPlaceCamRotations(int rot){
        inPlaceCamRotations_ = rot;
    }

    void setLightings(int lightings){
        lightings_ = lightings;
    }

    void setHeight(int num_heights, float height_step){
        heights_ = num_heights;
        height_step_ = height_step;
    }

  /* \brief Sets the size of the render window
   * \param res resolution size
   */
  void setResolution (int resX, int resY)
  {
    resolutionX_ = resX;
    resolutionY_ = resY;
    depthbuf_.resize(resolutionX_ * resolutionY_);
  }

  /* \brief Wether to use the vertices or triangle centers of the tesselated sphere
   * \param use true indicates to use vertices, false triangle centers
   */

  void
  setUseVertices (bool use)
  {
    use_vertices_ = use;
  }


  /* \brief How many times the icosahedron should be tesselated. Results in more or less camera positions and generated views.
   * \param level amount of tesselation
   */
  void
  setTesselationLevel (int level)
  {
    tesselation_level_ = level;
  }

  /* \brief Sets the view angle of the virtual camera
   * \param angle view angle in degrees
   */
  void
  setViewAngle (float angle)
  {
    view_angle_ = angle;
  }

  void setStartHeight(float h){
     start_height_ = h;
  }

  void setAboveZ(bool z){
      above_z_ = z;
  }

  void setBelowZ(bool z){
      below_z_ = z;
  }

  void setRenderAround0(bool r0) {
      render_around_0_ = r0;
  }

  void setObjectRadius(double r) {
      object_radius_ = r;
  }

  /* \brief performs the rendering and stores the generated information
   */
  void
  generateViews ();

 };


#endif /* RENDER_VIEWS_TESSELATED_SPHERE_H_ */
