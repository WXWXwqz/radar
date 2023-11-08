#include <fstream>
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::ifstream file("./tmp/matrix.bin", std::ios::binary);

    // 读取矩阵形状
    std::vector<int64_t> shape(3);
    file.read(reinterpret_cast<char*>(shape.data()), 3 * sizeof(int64_t));

    // 读取数据类型字符串
    std::string dtype_str;
    std::getline(file, dtype_str);

    // 根据读取的形状和数据类型创建矩阵
    // 这里假设矩阵的数据类型是double，您需要根据dtype_str来确定具体的数据类型
    std::vector<double> matrix(shape[0] * shape[1] * shape[2]);

    // 读取矩阵数据
    file.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(double));
    // 打印一些数据来验证
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                std::cout << matrix[i * shape[1] * shape[2] + j * shape[2] + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    file.close();
    return 0;
}
