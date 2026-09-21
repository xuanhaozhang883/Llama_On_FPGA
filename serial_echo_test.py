import importlib

try:
    serial = importlib.import_module("serial")
except ImportError as exc:
    raise ImportError(
        "未安装 pyserial，请先运行: python -m pip install pyserial"
    ) from exc


try:
    AutoTokenizer = importlib.import_module("transformers").AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "未安装 transformers，请先运行: pip install transformers"
    ) from exc

from token_protocol import build_packet, parse_packet


PORT = "COM4"
BAUDRATE = 115200
TIMEOUT = 3.0

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# 如果本机已经缓存了 tokenizer，可以设为 True。
# 如果没有缓存，需要联网下载，则改成 False。
LOCAL_ONLY = True


def main():
    # 1. 加载 Llama 3 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=LOCAL_ONLY,
    )

    # 2. 从电脑输入一句话
    text = input("请输入一句话：")

    # 3. 句子转换为 Llama Token ID
    input_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
    )

    print("Token IDs:", input_ids)

    # 4. Token ID列表打包成二进制数据包
    packet = build_packet(input_ids)

    print("发送字节数:", len(packet))
    print("发送数据:", packet.hex(" "))

    # 5. 打开开发板对应的串口
    with serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=TIMEOUT,
        write_timeout=TIMEOUT,
    ) as ser:

        # 清除之前残留的串口数据
        ser.reset_input_buffer()

        # 6. 发送完整数据包
        ser.write(packet)
        ser.flush()

        # 7. 等待 FPGA ARM 程序原样返回相同长度的数据
        received = ser.read(len(packet))

    print("接收字节数:", len(received))
    print("接收数据:", received.hex(" "))

    # 8. 首先比较二进制数据是否完全一致
    if received != packet:
        print("Echo失败：返回字节与发送字节不一致")
        return

    # 9. 再使用协议解析函数验证 checksum 和 Token ID
    received_ids = parse_packet(received)

    print("Echo成功")
    print("返回的Token IDs:", received_ids)
    print("Token ID是否一致:", received_ids == input_ids)

    # 10. 在电脑端把返回的 Token ID 解码回文字
    restored_text = tokenizer.decode(
        received_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("返回Token还原文字:", restored_text)


if __name__ == "__main__":
    main()