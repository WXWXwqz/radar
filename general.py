import numpy as np
def trans_npy_to_bin(npy_name,bin_name):
    # 假设我们有一个.npy文件
    matrix = np.load(npy_name)


    # 转换为二进制文件
    with open('matrix.bin', 'wb') as f:
        # 写入矩阵形状信息
        f.write(np.array(matrix.shape, dtype=np.int64).tobytes())

        # 将dtype作为字符串写入
        dtype_str = str(matrix.dtype)
        f.write(dtype_str.encode('ascii'))
        f.write(b'\n')  # 添加一个换行符作为dtype字符串的终止

        # 写入实际的矩阵数据
        f.write(matrix.tobytes())


if __name__ == "__main__":
    trans_npy_to_bin("normal.npy","data.bin")