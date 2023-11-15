import numpy as np
def trans_npy_to_bin(npy_name,bin_name):
    # ����������һ��.npy�ļ�
    matrix = np.load(npy_name)


    # ת��Ϊ�������ļ�
    with open('matrix.bin', 'wb') as f:
        # д�������״��Ϣ
        f.write(np.array(matrix.shape, dtype=np.int64).tobytes())

        # ��dtype��Ϊ�ַ���д��
        dtype_str = str(matrix.dtype)
        f.write(dtype_str.encode('ascii'))
        f.write(b'\n')  # ���һ�����з���Ϊdtype�ַ�������ֹ

        # д��ʵ�ʵľ�������
        f.write(matrix.tobytes())


if __name__ == "__main__":
    trans_npy_to_bin("normal.npy","data.bin")