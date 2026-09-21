#include "xparameters.h"
#include "xil_printf.h"
#include "xil_cache.h"

#include "platform.h"
#include "netif/xadapter.h"
#include "lwip/init.h"
#include "lwip/inet.h"
#include "lwip/ip_addr.h"
#include "lwip/err.h"
#include "lwip/tcp.h"

#include "echo_server.h"

/*
 * 你的 PS 配置截图显示 GEM3 已启用。
 * 不同 XSA 生成的宏名可能不同，所以优先使用 GEM3，
 * 若 BSP 给它编号为 0，则退回 XPAR_XEMACPS_0_BASEADDR。
 */
#if defined(XPAR_XEMACPS_3_BASEADDR)
#define ETH_BASEADDR XPAR_XEMACPS_3_BASEADDR
#elif defined(XPAR_XEMACPS_0_BASEADDR)
#define ETH_BASEADDR XPAR_XEMACPS_0_BASEADDR
#else
#error "XEMACPS base address not found; check the exported platform/XSA"
#endif

/* 电脑端设置为 192.168.10.100，开发板使用 192.168.10.10。 */
#define BOARD_IP0  192
#define BOARD_IP1  168
#define BOARD_IP2  10
#define BOARD_IP3  10

static struct netif server_netif;
/* Xilinx lwIP platform.c 用它周期性检查 PHY 链路状态。 */
struct netif *echo_netif;

/*
 * 这是一个本地管理的 MAC 地址。开发板若已有固定 MAC，
 * 应改成板卡手册提供的 MAC，避免和其他设备冲突。
 */
static unsigned char mac_address[] = {
    0x02, 0x00, 0x00, 0x10, 0x00, 0x10
};

/* Vitis 的 lwIP 示例模板通常在 platform.c 中驱动这两个标志位。 */
extern volatile int TcpFastTmrFlag;
extern volatile int TcpSlowTmrFlag;
extern void platform_enable_interrupts(void);

static void print_ip(const char *name, const ip_addr_t *addr)
{
    xil_printf("%s: %d.%d.%d.%d\r\n", name,
               ip4_addr1(ip_2_ip4(addr)),
               ip4_addr2(ip_2_ip4(addr)),
               ip4_addr3(ip_2_ip4(addr)),
               ip4_addr4(ip_2_ip4(addr)));
}

int main(void)
{
    ip_addr_t ipaddr;
    ip_addr_t netmask;
    ip_addr_t gateway;
    unsigned char *mac = mac_address;
    int status;

    init_platform();
    xil_printf("\r\n--- XCZU15EG Ethernet TCP Echo ---\r\n");

    echo_netif = &server_netif;

    IP4_ADDR(&ipaddr, BOARD_IP0, BOARD_IP1, BOARD_IP2, BOARD_IP3);
    IP4_ADDR(&netmask, 255, 255, 255, 0);
    IP4_ADDR(&gateway, 0, 0, 0, 0);

    print_ip("Board IP", &ipaddr);
    print_ip("Netmask", &netmask);

    lwip_init();

    if (xemac_add(echo_netif, &ipaddr, &netmask, &gateway,
                  mac, ETH_BASEADDR) == NULL) {
        xil_printf("xemac_add failed; check GEM3/PHY/MIO configuration\r\n");
        cleanup_platform();
        return -1;
    }

    netif_set_default(echo_netif);
    netif_set_up(echo_netif);

    platform_enable_interrupts();

    status = eth_echo_start();
    if (status != ERR_OK) {
        xil_printf("Echo server start failed: %d\r\n", status);
        cleanup_platform();
        return -1;
    }

    xil_printf("Connect PC to 192.168.10.10:5001\r\n");

    while (1) {
        if (TcpFastTmrFlag) {
            tcp_fasttmr();
            TcpFastTmrFlag = 0;
        }
        if (TcpSlowTmrFlag) {
            tcp_slowtmr();
            TcpSlowTmrFlag = 0;
        }

        /* 从 GEM 驱动取出收到的以太网帧，并交给 lwIP。 */
        xemacif_input(&server_netif);
    }

    cleanup_platform();
    return 0;
}
