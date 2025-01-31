// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    bool inTriangle = false;
    Eigen::Vector3f P = Vector3f(x, y, 1.0);
    Eigen::Vector3f AB= _v[1]-_v[0];
    Eigen::Vector3f BC= _v[2]-_v[1];
    Eigen::Vector3f CA= _v[0]-_v[2];

    Eigen::Vector3f AP= P-_v[0];
    Eigen::Vector3f BP= P-_v[1];
    Eigen::Vector3f CP= P-_v[2];

    float z1=AB.cross(AP).z();
    float z2=BC.cross(BP).z();
    float z3=CA.cross(CP).z();

    if((z1>0 && z2>0 && z3>0)|| (z1<0 && z2<0 && z3<0)){
        inTriangle = true;
    }

    return inTriangle;
    
}


static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}//计算三角形的重心坐标

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];
      // 获取顶点缓冲区、索引缓冲区和颜色缓冲区的引用

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;
    // 计算视口变换参数

    // 计算模型视图投影矩阵（MVP矩阵）
    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    int min_x = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    int max_x = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    int min_y = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    int max_y = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));

// 如果是这样，请使用以下代码获取插值 z 值。
// auto[alpha， beta， gamma] = computeBarycentric2D（x， y， t.v）;
// float w_reciprocal = 1.0/（alpha / v[0].w（） + beta / v[1].w（） + gamma / v[2].w（））;
// float z_interpolated = alpha * v[0].z（） / v[0].w（） + beta * v[1].z（） / v[1].w（） + gamma * v[2].z（） / v[2].w（）;
// z_interpolated *= w_reciprocal;
    bool msaa = true;
    if(!msaa)
    {
        for (int x =(int) min_x; x < max_x; x++)
        {
            for (int y =(int) min_y; y < max_y; y++)
            {
                if (insideTriangle(x+0.5, y+0.5, t.v))
                {
                    auto[alpha, beta, gamma] = computeBarycentric2D(x, y,t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    float buf_index = get_index(x, y);
                    if (z_interpolated < depth_buf[buf_index])
                    {
                        depth_buf[buf_index] = z_interpolated;
                        set_pixel(Eigen::Vector3f(x, y, z_interpolated), t.getColor());
                    }
                }
            }
        }
    }
    else{
        for (int x =(int) min_x; x < max_x; x++)
        {
            for (int y =(int) min_y; y < max_y; y++)
            {
                int sample_point= 0;
                for(int i=0; i<2; i++){
                    for(int j=0; j<2; j++){
                        float x_sample =(float)x + 0.25+0.5*i;
                        float y_sample =(float)y + 0.25+0.5*j;
                        if (insideTriangle(x_sample, y_sample, t.v))
                        {
                            auto[alpha, beta, gamma] = computeBarycentric2D(x_sample, y_sample,t.v);
                            float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                            z_interpolated *= w_reciprocal;

                            int buf_index = (x*2+i)+(y*2+j)*width*2;
                            if(z_interpolated < depth_buf_msaa22[buf_index])
                            {
                                depth_buf_msaa22[buf_index] = z_interpolated;
                                sample_point++;
                            }
                        }
                    }

                }
                if(sample_point>0)
                {
                    float sample = float(sample_point)/4.0f;
                    msaa_pixel(Eigen::Vector3f(x, y, 1.0f), t.getColor()*sample);
                }


            }
        }



    }

// TODO ：如果应该绘制三角形，请将当前像素（使用 set_pixel 函数）设置为三角形的颜色（使用 getColor 函数）。

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(depth_buf_msaa22.begin(), depth_buf_msaa22.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    depth_buf_msaa22.resize(w * h * 2 * 2);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;

}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

void rst::rasterizer::msaa_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    // 两三角形边界处应该是颜色的混合
    // 题目里边界处就是3/4的绿色混合1/4的蓝色
    frame_buf[ind] += color;
}

// clang-format on