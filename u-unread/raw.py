# // Loop 1: 计算 B[i]
# for (i = 0; i < N; i++)
#     B[i] = A[i] * 2;
#
# // Loop 2: 读 B[i+1]（超前读）
# for (i = 0; i < N-1; i++)
#     C[i] = B[i+1] + B[i];
# for (i = 0; i < N-1; i++) {
#     B[i]   = A[i] * 2;
# // B[i+1] 尚未计算!
# C[i]   = B[i+1] + B[i];
# // ^ 读到旧值 / 未初始化
# }

def main():
    N = 10  # 示例大小
    A = [i for i in range(N)]  # 初始化数组 A
    B = [0] * N  # 初始化数组 B
    C = [0] * (N - 1)  # 初始化数组 C

    # Loop 1: 计算 B[i]
    for i in range(N):
        B[i] = A[i] * 2
    print("B数组:", B)

    # Loop 2: 读 B[i+1]（超前读）
    for i in range(N - 1):
        C[i] = B[i + 1] + B[i]
    print("C数组:", C)

    print('=' * 50)
    B = [0] * N  # 初始化数组 B
    C = [0] * (N - 1)  # 初始化数组 C
    # 这个循环存在数据依赖问题：在计算B[i]的同时读取B[i+1]
    for i in range(N-1):
        B[i] = A[i] * 2
        C[i] = B[i+1] + B[i]
    print("B数组:", B)
    print("C数组:", C)

if __name__ == "__main__":
    main()
