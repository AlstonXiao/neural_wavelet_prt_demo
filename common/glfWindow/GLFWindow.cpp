// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "GLFWindow.h"
#include <iomanip>
#include <sstream>
#include <vector>

namespace nert_renderer {
  using namespace gdt;
  
  static void glfw_error_callback(int error, const char* description)
  {
    fprintf(stderr, "Error: %s\n", description);
  }
  
  GLFWindow::~GLFWindow()
  {
    glfwDestroyWindow(handle);
    glfwTerminate();
  }

  GLFWindow::GLFWindow(const std::string &title, const vec2i size)
  {
    glfwSetErrorCallback(glfw_error_callback);
    // glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
      
    if (!glfwInit())
      exit(EXIT_FAILURE);
      
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
      
    handle = glfwCreateWindow(size.x, size.y, title.c_str(), NULL, NULL);
    if (!handle) {
      glfwTerminate();
      exit(EXIT_FAILURE);
    }
      
    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);
    glfwSwapInterval( 0 );
  }

  /*! callback for a window resizing event */
  static void glfwindow_reshape_cb(GLFWwindow* window, int width, int height )
  {
    GLFWindow *gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->resize(vec2i(width,height));
  // assert(GLFWindow::current);
  //   GLFWindow::current->resize(vec2i(width,height));
  }

  /*! callback for a key press */
  static void glfwindow_key_cb(GLFWwindow *window, int key, int scancode, int action, int mods) 
  {
    GLFWindow *gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->key(key,mods,action);

  }

  /*! callback for _moving_ the mouse to a new position */
  static void glfwindow_mouseMotion_cb(GLFWwindow *window, double x, double y) 
  {
    GLFWindow *gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->mouseMotion(vec2i((int)x, (int)y));
  }

  /*! callback for pressing _or_ releasing a mouse button*/
  static void glfwindow_mouseButton_cb(GLFWwindow *window, int button, int action, int mods) 
  {
    GLFWindow *gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);
    // double x, y;
    // glfwGetCursorPos(window,&x,&y);
    gw->mouseButton(button,action,mods);
  }
  
  void GLFWindow::run()
  {
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(vec2i(width,height));

    // glfwSetWindowUserPointer(window, GLFWindow::current);
    glfwSetFramebufferSizeCallback(handle, glfwindow_reshape_cb);
    glfwSetMouseButtonCallback(handle, glfwindow_mouseButton_cb);
    glfwSetKeyCallback(handle, glfwindow_key_cb);
    glfwSetCursorPosCallback(handle, glfwindow_mouseMotion_cb);
    
    double previousTime = 0;
    double currentTime = 0;
    double diff;
    unsigned int counter = 0;
    unsigned int globalCounter = 0;
    std::vector<float> averageMS(100);

    while (!glfwWindowShouldClose(handle)) {
        currentTime = glfwGetTime();
        diff = currentTime - previousTime;
      render(diff);
      draw();
      
      counter++;
      
      if (diff > (1. / 30.)) {
          std::stringstream stream;
          stream << std::round((1. / diff) * counter);
          std::string FPS = stream.str();

          std::stringstream stream_1;
          stream_1 << std::fixed << std::setprecision(2) << (diff / counter * 1000.);
          std::string ms = stream_1.str();

          std::string newTitle = "NERT - " + FPS + " FPS / " + ms + " ms";
          glfwSetWindowTitle(handle, newTitle.c_str());
          averageMS[globalCounter % 100] = diff / counter * 1000.;
          globalCounter++;
          previousTime = currentTime;
          counter = 0;
      }
      if (globalCounter % 20 == 0) {
          float sum = 0;
          unsigned int total = std::min(globalCounter, (unsigned int)100);
          for (auto i = 0; i < total; i++) {
              sum += averageMS[i];
          }
          sum = sum / 100;
          // std::cout << "Average ms is " << sum << ". Average FPS is: " << 1 / sum * 1000 << std::endl;
      }
      glfwSwapBuffers(handle);
      glfwPollEvents();
    }
  }

  // GLFWindow *GLFWindow::current = nullptr;
  
} // ::nert_renderer
