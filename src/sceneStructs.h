#ifndef SCENE_STRUCTS_H
#define SCENE_STRUCTS_H

//This is a lighter weight version of obj
struct Mesh {
  int vbosize;
  int nbosize;
  int cbosize;
  int ibosize;
  float* vbo;
  float* nbo;
  float* cbo;
  int* ibo;
};



#endif