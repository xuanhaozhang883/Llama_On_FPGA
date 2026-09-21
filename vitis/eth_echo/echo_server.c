#include "echo_server.h"

#include "xil_printf.h"
#include "lwip/err.h"
#include "lwip/ip_addr.h"
#include "lwip/tcp.h"

#define ETH_ECHO_PORT 5001

/*
 * TCP 是字节流，不保证一次 recv() 就对应电脑端的一次 send()。
 * 因此这里按 pbuf 链逐段回显，而不是假设每次只收到一个完整包。
 */
static err_t echo_recv(void *arg, struct tcp_pcb *tpcb,
                       struct pbuf *p, err_t err)
{
    struct pbuf *q;
    err_t write_err = ERR_OK;

    (void)arg;

    if (p == NULL) {
        /* 对端正常关闭 TCP 连接。 */
        tcp_close(tpcb);
        return ERR_OK;
    }

    if (err != ERR_OK) {
        pbuf_free(p);
        return err;
    }

    /* 告诉 TCP 接收窗口：这些字节已经由应用取走。 */
    tcp_recved(tpcb, p->tot_len);

    for (q = p; q != NULL; q = q->next) {
        if (q->len == 0U) {
            continue;
        }

        /* TCP_WRITE_FLAG_COPY：lwIP 会复制数据，pbuf 释放后仍可发送。 */
        write_err = tcp_write(tpcb, q->payload, q->len,
                              TCP_WRITE_FLAG_COPY);
        if (write_err != ERR_OK) {
            xil_printf("tcp_write failed: %d\r\n", write_err);
            break;
        }
    }

    if (write_err == ERR_OK) {
        write_err = tcp_output(tpcb);
    }

    pbuf_free(p);
    return write_err;
}

static void echo_error(void *arg, err_t err)
{
    (void)arg;
    xil_printf("TCP connection error: %d\r\n", err);
}

static err_t echo_accept(void *arg, struct tcp_pcb *newpcb, err_t err)
{
    (void)arg;

    if (err != ERR_OK || newpcb == NULL) {
        return (err == ERR_OK) ? ERR_VAL : err;
    }

    tcp_arg(newpcb, NULL);
    tcp_recv(newpcb, echo_recv);
    tcp_err(newpcb, echo_error);

    xil_printf("TCP client connected\r\n");
    return ERR_OK;
}

err_t eth_echo_start(void)
{
    struct tcp_pcb *listen_pcb;
    err_t err;

    listen_pcb = tcp_new_ip_type(IPADDR_TYPE_V4);
    if (listen_pcb == NULL) {
        xil_printf("tcp_new failed\r\n");
        return ERR_MEM;
    }

    err = tcp_bind(listen_pcb, IP_ADDR_ANY, ETH_ECHO_PORT);
    if (err != ERR_OK) {
        xil_printf("tcp_bind failed: %d\r\n", err);
        tcp_close(listen_pcb);
        return err;
    }

    listen_pcb = tcp_listen(listen_pcb);
    if (listen_pcb == NULL) {
        xil_printf("tcp_listen failed\r\n");
        return ERR_MEM;
    }

    tcp_accept(listen_pcb, echo_accept);
    xil_printf("TCP Echo listening on port %d\r\n", ETH_ECHO_PORT);
    return ERR_OK;
}
