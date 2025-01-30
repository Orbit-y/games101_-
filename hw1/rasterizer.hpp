//
// Created by goksu on 4/6/19.
//

#pragma once

#include "Triangle.hpp"
#include <algorithm>
#include <eigen3/Eigen/Eigen>
using namespace Eigen;

namespace rst {
    // 枚举类型 Buffers，表示缓冲区类型
enum class Buffers
{
     Color = 1,  // 颜色缓冲区
    Depth = 2   // 深度缓冲区
};

inline Buffers operator|(Buffers a, Buffers b)
{// 重载 | 运算符，用于组合缓冲区类型
    return Buffers((int)a | (int)b);
}

inline Buffers operator&(Buffers a, Buffers b)
{// 重载 | 运算符，用于组合缓冲区类型
    return Buffers((int)a & (int)b);
}

// 枚举类型 Primitive，表示支持的图元类型
enum class Primitive
{
    Line,      // 线段
    Triangle   // 三角形
};

/*
 * For the curious : The draw function takes two buffer id's as its arguments.
 * These two structs make sure that if you mix up with their orders, the
 * compiler won't compile it. Aka : Type safety
 * */
struct pos_buf_id
{
    int pos_id = 0;// 位置缓冲区的 ID
};

struct ind_buf_id
{
    int ind_id = 0;// 索引缓冲区的 ID
};

class rasterizer// 光栅化器类
{
  public:
    rasterizer(int w, int h); // 构造函数，初始化光栅化器的宽度和高度
    
     // 加载顶点位置数据，返回位置缓冲区的 ID
    pos_buf_id load_positions(const std::vector<Eigen::Vector3f>& positions);
    
    // 加载索引数据，返回索引缓冲区的 ID
    ind_buf_id load_indices(const std::vector<Eigen::Vector3i>& indices);

    //设置模型矩阵,设置视图矩阵,设置投影矩阵
    void set_model(const Eigen::Matrix4f& m);
    void set_view(const Eigen::Matrix4f& v);
    void set_projection(const Eigen::Matrix4f& p);

    //设置帧缓冲区中指定像素的颜色
    void set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color);

    // 清空指定的缓冲区（颜色缓冲区或深度缓冲区）
    void clear(Buffers buff);

    // 绘制图元，根据位置缓冲区和索引缓冲区的 ID 以及图元类型
    void draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, Primitive type);

    // 返回帧缓冲区的引用
    std::vector<Eigen::Vector3f>& frame_buffer() { return frame_buf; }

  private:
    void draw_line(Eigen::Vector3f begin, Eigen::Vector3f end);// 绘制线段
    void rasterize_wireframe(const Triangle& t); // 光栅化三角形的线框

  private:
    Eigen::Matrix4f model;
    Eigen::Matrix4f view;
    Eigen::Matrix4f projection;

    std::map<int, std::vector<Eigen::Vector3f>> pos_buf;
    std::map<int, std::vector<Eigen::Vector3i>> ind_buf;

    std::vector<Eigen::Vector3f> frame_buf;
    std::vector<float> depth_buf;
    int get_index(int x, int y);

    int width, height;

    int next_id = 0;
    int get_next_id() { return next_id++; }
};
} // namespace rst
