# // Loop 1: 读 A[i+1]（超前读）
# for (i = 0; i < N-1; i++)
#     B[i] = A[i+1] + 1;
#
# // Loop 2: 写 A[i]（覆盖写）
# for (i = 0; i < N; i++)
#     A[i] = C[i] * 3;
#
#
# for (i = 0; i < N-1; i++) {
#     B[i] = A[i+1] + 1;
# // 先读 A[i+1]，没问题
# A[i] = C[i] * 3;
# // ^ 写 A[i]，下次 B[i-1] 读
# // A[i] 已被覆盖！
# }

# Loop 1: 读 A[i+1]（超前读）
def loop1(N, A, B):
    for i in range(N-1):
        B[i] = A[i+1] + 1

# Loop 2: 写 A[i]（覆盖写）
def loop2(N, A, C):
    for i in range(N):
        A[i] = C[i] * 3

# 有问题的循环：WAR (Write-After-Read) 依赖
def problematic_loop(N, A, B, C):
    for i in range(N-1):
        B[i] = A[i+1] + 1
        # 先读 A[i+1]，没问题
        A[i] = C[i] * 3
        # ^ 写 A[i]，下次 B[i-1] 读
        # A[i] 已被覆盖！

def main():
    N = 10  # 示例大小
    A = [i for i in range(N)]  # 初始化数组 A
    B = [0] * N  # 初始化数组 B
    C = [i * 2 for i in range(N)]  # 初始化数组 C

    print("初始 A数组:", A)
    print("初始 C数组:", C)

    # 正确的做法：先读后写
    for i in range(N):
        B[i] = A[i] + 1
    print("Loop 1 后 B数组:", B)

    for i in range(N-1):
        A[i+1] = C[i] * 3
    print("Loop 2 后 A数组:", A)

    print("=" * 50)

    # 重置数组
    A = [i for i in range(N)]
    B = [0] * (N)
    C = [i * 2 for i in range(N)]

    print("重置后 A数组:", A)

    # 执行有问题的循环
    for i in range(N-1):
        B[i] = A[i] + 1
        # 先读 A[i+1]，没问题
        A[i+1] = C[i] * 3
    print("问题循环后 B数组:", B)
    print("问题循环后 A数组:", A)

if __name__ == "__main__":
    main()
