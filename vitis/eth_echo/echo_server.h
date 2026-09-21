#ifndef ETH_ECHO_SERVER_H
#define ETH_ECHO_SERVER_H

#include "lwip/tcp.h"

/* 在 TCP 5001 端口上启动一个原样返回数据的 Echo 服务。 */
err_t eth_echo_start(void);

#endif
