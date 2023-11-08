#include <fstream>
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::ifstream file("./tmp/matrix.bin", std::ios::binary);

    // ��ȡ������״
    std::vector<int64_t> shape(3);
    file.read(reinterpret_cast<char*>(shape.data()), 3 * sizeof(int64_t));

    // ��ȡ���������ַ���
    std::string dtype_str;
    std::getline(file, dtype_str);

    // ���ݶ�ȡ����״���������ʹ�������
    // ���������������������double������Ҫ����dtype_str��ȷ���������������
    std::vector<double> matrix(shape[0] * shape[1] * shape[2]);

    // ��ȡ��������
    file.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(double));
    // ��ӡһЩ��������֤
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
