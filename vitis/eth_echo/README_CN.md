# eth_echo：最小 TCP 以太网回显程序

## 作用

这个程序不是 Attention 程序，也不传 QKV。它只验证：

```text
电脑 TCP 客户端 -> 开发板 PS GEM3 -> lwIP -> 原样返回电脑
```

开发板静态 IP：`192.168.10.10`

TCP 端口：`5001`

电脑有线网卡建议设置为：`192.168.10.100/24`，默认网关和 DNS 留空。

## 在 Vitis 2025.2 中的正确起点

当前工作区中的 `eth_echo` 是 **Empty Application**，不会自动生成
`platform.c`、`platform.h` 等 lwIP 示例辅助文件。**不要把本目录的
三个 C 文件直接放进空应用编译**；它们依赖示例工程的网络平台代码。

建议在 Vitis 中打开 `View > Examples`，搜索 `lwIP Echo Server`，
选择 `Create Application Component from Template`，用一个新名称如
`eth_echo_ref` 创建，选择现有的 `platform` 和
`standalone_psu_cortexa53_0`。先运行官方示例并确认板端网卡和 PHY
能工作，再参考本目录的 `echo_server.c` 学习回调和端口修改。

`lwipopts.h` 通常属于启用 lwIP 后的平台/BSP 配置，并不一定显示在
Application 的 `Sources/src` 树中。不要为了补齐文件名手工创建空文件。

## 注意

`XPAR_XEMACPS_3_BASEADDR` 是 GEM3 的常见宏名，但最终以生成的
`platform/include/xparameters.h` 为准。程序同时兼容退回到
`XPAR_XEMACPS_0_BASEADDR` 的情况。

在你当前导出的 platform 中，检查到实际生成的是
`XPAR_XEMACPS_0_BASEADDR`。这不代表 GEM0 被启用，而是因为只有一个
GEM 实例被导出，驱动把它编号为实例 0。

板卡如果有手册指定的 MAC 地址，应把 `main.c` 中的示例 MAC 换成板卡 MAC。
