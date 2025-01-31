//
// Created by LEI XU on 4/11/19.
//

#include "Triangle.hpp"
#include <algorithm>
#include <array>


Triangle::Triangle() {
    v[0] << 0,0,0;
    v[1] << 0,0,0;
    v[2] << 0,0,0;

    color[0] << 0.0, 0.0, 0.0;
    color[1] << 0.0, 0.0, 0.0;
    color[2] << 0.0, 0.0, 0.0;

    tex_coords[0] << 0.0, 0.0;
    tex_coords[1] << 0.0, 0.0;
    tex_coords[2] << 0.0, 0.0;
}

void Triangle::setVertex(int ind, Vector3f ver){
    v[ind] = ver;
}//设置三角形的顶点
void Triangle::setNormal(int ind, Vector3f n){
    normal[ind] = n;
}//这段代码的功能是设置三角形的法线向量。
void Triangle::setColor(int ind, float r, float g, float b) {
    if((r<0.0) || (r>255.) ||
       (g<0.0) || (g>255.) ||
       (b<0.0) || (b>255.)) {
        fprintf(stderr, "ERROR! Invalid color values");
        fflush(stderr);
        exit(-1);
    }

    color[ind] = Vector3f((float)r/255.,(float)g/255.,(float)b/255.);
    return;
}//设置三角形的颜色
void Triangle::setTexCoord(int ind, float s, float t) {
    tex_coords[ind] = Vector2f(s,t);
}//设置三角形顶点的纹理坐标。输入参数包括顶点索引ind和纹理坐标s、t，将这些坐标存储在tex_coords数组中对应的位置。

std::array<Vector4f, 3> Triangle::toVector4() const
{
    std::array<Eigen::Vector4f, 3> res;
    std::transform(std::begin(v), std::end(v), res.begin(), [](auto& vec) { return Eigen::Vector4f(vec.x(), vec.y(), vec.z(), 1.f); });
    return res;
}//将三角形的三个顶点转换为齐次坐标（x, y, z, 1）
