#ifndef ETH_ECHO_PLATFORM_CONFIG_H
#define ETH_ECHO_PLATFORM_CONFIG_H

/*
 * 当前导出的 xparameters.h 将“唯一启用的 GEM3”编号为实例 0，
 * 因此这里使用 XPAR_XEMACPS_0_BASEADDR。最终以 Vitis 生成的
 * xparameters.h 为准，不要按 GEM3 的名字强行写成 _3。
 */
#include "xparameters.h"

#define PLATFORM_EMAC_BASEADDR XPAR_XEMACPS_0_BASEADDR

#endif
