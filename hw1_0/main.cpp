#include<cmath>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<iostream>

int main()
{
    float x,y;
    std::cin>>x>>y;
    Eigen::Vector3f v(x,y,1.0f);
    std::cout<<"rotate angle"<<std::endl;
    float angle_rotate;
    std::cin>>angle_rotate;
    Eigen::Matrix3f rotate_matrix;
    rotate_matrix<<std::cos(angle_rotate/180.0*acos(-1)),std::sin(-angle_rotate/180.0*acos(-1)),0.0f,std::sin(angle_rotate/180.0*acos(-1)),std::cos(angle_rotate/180.0*acos(-1)),0,0,0,1;
    std::cout<<"translation"<<std::endl;
    float tx,ty;
    std::cin>>tx>>ty;
    Eigen::Matrix3f translation_matrix;
    translation_matrix<<1.0f,0.0f,tx,0.0f,1.0f,ty,0.0f,0.0f,1.0f;
    std::cout<<translation_matrix*rotate_matrix*v<<std::endl;
    return 0;

}