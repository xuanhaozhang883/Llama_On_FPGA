# Attention TCP v1协议

本文是Llama第0层PC–PS通信的唯一字节级规范。Python和C实现必须同时满足本文及固定测试向量。

## 1. 连接和编码

- TCP端口：`5001`。
- 同一连接完成一笔请求和对应响应。
- v1只允许一个客户端、一个请求在途。
- 魔数：ASCII `FPTA`，对应字节`46 50 54 41`。
- 所有多字节整数均为无符号小端编码。
- CRC算法：`zlib.crc32(payload) & 0xffffffff`。
- CRC只覆盖对应载荷，不覆盖头部。
- C端必须逐字段读写，不能把可能含有填充字节的结构体直接发送到网络。

## 2. 固定张量载荷

| 张量 | 类型与布局 | 字节数 |
|---|---|---:|
| Q | BF16小端`[32,128,128]` | 1,048,576 |
| K | BF16小端`[8,128,128]` | 262,144 |
| V | BF16小端`[8,128,128]` | 262,144 |
| Context | BF16小端`[32,128,128]` | 1,048,576 |

请求载荷固定为`Q || K || V`，共1,572,864字节。PC发送RoPE前Q/K/V。`valid_tokens`表示前`L`行有效，第`L`行及以后必须补零。

成功响应始终返回完整1 MiB Context。Context是`o_proj`之前的数据，布局为`[head][token][dim]`。

## 3. 请求头

请求头固定40字节：

| 偏移 | 字段 | 类型 | 约束 |
|---:|---|---|---|
| 0 | magic | 4字节 | `FPTA` |
| 4 | version | u8 | `1` |
| 5 | command | u8 | `1`，`RUN_ATTENTION` |
| 6 | header_bytes | u16 | `40` |
| 8 | request_id | u32 | 由PC生成 |
| 12 | valid_tokens | u16 | `1..128` |
| 14 | layer_index | u16 | v1固定为`0` |
| 16 | q_bytes | u32 | `1,048,576` |
| 20 | k_bytes | u32 | `262,144` |
| 24 | v_bytes | u32 | `262,144` |
| 28 | q_crc32 | u32 | 只覆盖Q |
| 32 | k_crc32 | u32 | 只覆盖K |
| 36 | v_crc32 | u32 | 只覆盖V |

Python格式：

```python
struct.Struct("<4sBBHIHHIIIIII")
```

## 4. 响应头

响应头固定32字节：

| 偏移 | 字段 | 类型 | 约束 |
|---:|---|---|---|
| 0 | magic | 4字节 | `FPTA` |
| 4 | version | u8 | `1` |
| 5 | command | u8 | `2`，`ATTENTION_RESULT` |
| 6 | header_bytes | u16 | `32` |
| 8 | request_id | u32 | 回显请求ID；无法解析时为0 |
| 12 | status_code | u16 | 见状态码表 |
| 14 | valid_tokens | u16 | 可确认时回显；否则为0 |
| 16 | context_bytes | u32 | 成功为1,048,576；错误为0 |
| 20 | context_crc32 | u32 | 成功时覆盖完整Context；错误为0 |
| 24 | fpga_status | u32 | FPGA原始状态；未启动时为0 |
| 28 | detail_code | u32 | 成功为0；错误时由模块定义详情 |

Python格式：

```python
struct.Struct("<4sBBHIHHIIII")
```

## 5. 状态码

| 值 | 名称 | 含义 |
|---:|---|---|
| 0 | `OK` | 成功，后跟完整Context |
| 1 | `BAD_MAGIC` | Magic错误 |
| 2 | `BAD_VERSION` | Version错误 |
| 3 | `BAD_HEADER` | Command、头部长度或其他头字段错误 |
| 4 | `BAD_LENGTH` | Q/K/V或Context长度错误 |
| 5 | `BAD_VALID_TOKENS` | Token数不在允许范围 |
| 6 | `BAD_LAYER_INDEX` | v1不是第0层 |
| 7 | `BAD_CRC` | 任一输入载荷CRC错误 |
| 8 | `BUSY` | 已有连接或请求在途 |
| 9 | `RESET_TIMEOUT` | Reset未在时限内完成 |
| 10 | `RUN_TIMEOUT` | PL运行超时 |
| 11 | `FPGA_ERROR` | FPGA错误或完成状态非法 |
| 12 | `INTERNAL_ERROR` | 服务器内部错误 |

错误响应只包含32字节响应头，不能附带Context。

## 6. 接收与错误边界

- TCP没有消息边界，接收端必须支持头部和载荷被任意拆分或合并。
- 只有完整接收Q/K/V、固定长度正确且三个CRC全部通过后，才能启动PL。
- 可以证明PL尚未启动的错误，只有在当前帧完整消费、下一帧边界明确时才能复用连接；否则关闭连接。
- Start已发出，或不能排除已发出后发生超时或硬件错误，服务器进入`FAULT`。
- `FAULT`禁止新请求和共享DDR复用；断线重连不能清除故障。
- 只有经过验证的系统复位，并确认PL/AXI不再访问共享DDR后，才能恢复服务。

## 7. 固定测试向量

CRC标准向量：

```text
输入ASCII：123456789
CRC32：cbf43926
```

请求头测试值：

```text
request_id   = 0x12345678
valid_tokens = 3
layer_index  = 0
q_crc32      = 0x11223344
k_crc32      = 0xaabbccdd
v_crc32      = 0x01020304
```

对应40字节请求头：

```text
46505441010128007856341203000000
00001000000004000000040044332211
ddccbbaa04030201
```

成功响应测试值：

```text
request_id   = 0x12345678
status_code  = 0
valid_tokens = 3
context_crc  = 0xdeadbeef
fpga_status  = 0x0000006d
detail_code  = 0
```

对应32字节响应头：

```text
46505441010220007856341200000300
00001000efbeadde6d00000000000000
```

