"""PC 端 TCP Echo 测试。

先运行开发板上的 eth_echo.elf，再运行本脚本。
"""

import socket
import time

BOARD_IP = "192.168.10.10"
BOARD_PORT = 5001


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(min(64 * 1024, size - len(data)))
        if not chunk:
            raise RuntimeError("开发板提前关闭了 TCP 连接")
        data.extend(chunk)
    return bytes(data)


def main() -> None:
    print(f"连接 {BOARD_IP}:{BOARD_PORT} ...")
    with socket.create_connection((BOARD_IP, BOARD_PORT), timeout=5.0) as sock:
        message = b"HELLO FPGA\r\n"
        sock.sendall(message)
        echo = recv_exact(sock, len(message))
        print("发送：", message)
        print("接收：", echo)
        print("短消息 Echo：", echo == message)

        payload = bytes(index & 0xFF for index in range(1024 * 1024))
        start = time.perf_counter()
        sock.sendall(payload)
        returned = recv_exact(sock, len(payload))
        elapsed = time.perf_counter() - start
        print("1 MiB Echo：", returned == payload)
        print(f"往返耗时：{elapsed:.3f} s")
        if elapsed > 0:
            print(f"往返有效吞吐：{len(payload) / elapsed / 1024 / 1024:.2f} MiB/s")


if __name__ == "__main__":
    main()
