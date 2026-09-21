"""教学版 token 回传协议：只构造/解析完整包，不负责串口收发。

包格式：AA 55 | command:u8 | count:u16le | IDs:u32le[count] | sum:u16le
校验覆盖包头到 payload 的全部字节，不包含校验字段自身。
累加校验不能检测所有错误；它不是 CRC，也不是安全认证。
"""

import struct

MAGIC = b"\xAA\x55"
CMD_ECHO = 0x01
MAX_INPUT_TOKENS = 128
VOCAB_SIZE = 128256  # 本练习固定使用原版 Llama 3-8B 的 ID 范围。


def build_packet(input_ids):
    """发送端：将非空 ID 列表编码为一个完整 bytes 数据包。"""
    count = len(input_ids)
    if not 1 <= count <= MAX_INPUT_TOKENS:
        raise ValueError("token 数量必须在 1～128 之间（包含 BOS）")
    for token_id in input_ids:
        if type(token_id) is not int or not 0 <= token_id < VOCAB_SIZE:
            raise ValueError("token ID 必须是 0～128255 范围内的整数")

    # <：小端；B：1 字节无符号数/用于命令；H：2 字节无符号数/用于Count。
    header = MAGIC + struct.pack("<BH", CMD_ECHO, count)
    payload = struct.pack(f"<{count}I", *input_ids)
    body = header + payload

    # 遍历 bytes 时，每个元素是 0～255 的整数。取累加结果的低 16 位。
    checksum = sum(body) & 0xFFFF
    packet = body + struct.pack("<H", checksum)
    return packet


def parse_packet(packet):
    """接收端模拟：检查一个完整包，再返回 ID 列表。

    这里只接受恰好一包数据；串口的分批接收、超时、重新同步以后再做。
    """
    if len(packet) < 7:  # 固定头 5 字节 + 校验 2 字节。
        raise ValueError("数据包太短，固定字段不完整")
    if packet[:2] != MAGIC:
        raise ValueError("包头错误，应为 AA 55")

    # 字节 0～1 是包头；字节 2 是命令；字节 3～4 是数量。
    command, count = struct.unpack("<BH", packet[2:5])
    if command != CMD_ECHO:
        raise ValueError("不支持的命令，当前只支持 01 回传测试")
    if not 1 <= count <= MAX_INPUT_TOKENS:
        raise ValueError("收到的 token 数量超出 1～128 范围")

    expected_length = 5 + count * 4 + 2
    if len(packet) != expected_length:
        raise ValueError("实际包长度与 Count 不匹配")

    # 最后 2 字节是对方发来的校验值；前面所有字节用于重新计算。
    received_checksum = struct.unpack("<H", packet[-2:])[0]
    calculated_checksum = sum(packet[:-2]) & 0xFFFF
    if received_checksum != calculated_checksum:
        raise ValueError("校验失败，拒绝接受数据包")

    payload = packet[5:-2]
    received_ids = list(struct.unpack(f"<{count}I", payload))
    if any(token_id >= VOCAB_SIZE for token_id in received_ids):
        raise ValueError("收到的 token ID 超出 Llama 3-8B 词表范围")
    return received_ids
