#ifndef FPT_ATTENTION_PROTOCOL_H
#define FPT_ATTENTION_PROTOCOL_H

#include <stdint.h>

/* TCP v1 wire constants. Multi-byte fields are always little-endian. */
#define FPT_MAGIC_BYTE_0 70U  /* F */
#define FPT_MAGIC_BYTE_1 80U  /* P */
#define FPT_MAGIC_BYTE_2 84U  /* T */
#define FPT_MAGIC_BYTE_3 65U  /* A */

#define FPT_PROTOCOL_VERSION 1U
#define FPT_COMMAND_RUN_ATTENTION 1U
#define FPT_COMMAND_ATTENTION_RESULT 2U

#define FPT_REQUEST_HEADER_BYTES 40U
#define FPT_RESPONSE_HEADER_BYTES 32U

#define FPT_Q_BYTES 1048576U
#define FPT_K_BYTES 262144U
#define FPT_V_BYTES 262144U
#define FPT_REQUEST_PAYLOAD_BYTES 1572864U
#define FPT_CONTEXT_BYTES 1048576U

#define FPT_MIN_VALID_TOKENS 1U
#define FPT_MAX_VALID_TOKENS 128U
#define FPT_LAYER_INDEX_V1 0U

/* Request header offsets. */
#define FPT_REQ_OFF_MAGIC 0U
#define FPT_REQ_OFF_VERSION 4U
#define FPT_REQ_OFF_COMMAND 5U
#define FPT_REQ_OFF_HEADER_BYTES 6U
#define FPT_REQ_OFF_REQUEST_ID 8U
#define FPT_REQ_OFF_VALID_TOKENS 12U
#define FPT_REQ_OFF_LAYER_INDEX 14U
#define FPT_REQ_OFF_Q_BYTES 16U
#define FPT_REQ_OFF_K_BYTES 20U
#define FPT_REQ_OFF_V_BYTES 24U
#define FPT_REQ_OFF_Q_CRC32 28U
#define FPT_REQ_OFF_K_CRC32 32U
#define FPT_REQ_OFF_V_CRC32 36U

/* Response header offsets. */
#define FPT_RSP_OFF_MAGIC 0U
#define FPT_RSP_OFF_VERSION 4U
#define FPT_RSP_OFF_COMMAND 5U
#define FPT_RSP_OFF_HEADER_BYTES 6U
#define FPT_RSP_OFF_REQUEST_ID 8U
#define FPT_RSP_OFF_STATUS_CODE 12U
#define FPT_RSP_OFF_VALID_TOKENS 14U
#define FPT_RSP_OFF_CONTEXT_BYTES 16U
#define FPT_RSP_OFF_CONTEXT_CRC32 20U
#define FPT_RSP_OFF_FPGA_STATUS 24U
#define FPT_RSP_OFF_DETAIL_CODE 28U

typedef enum {
    FPT_STATUS_OK = 0,
    FPT_STATUS_BAD_MAGIC = 1,
    FPT_STATUS_BAD_VERSION = 2,
    FPT_STATUS_BAD_HEADER = 3,
    FPT_STATUS_BAD_LENGTH = 4,
    FPT_STATUS_BAD_VALID_TOKENS = 5,
    FPT_STATUS_BAD_LAYER_INDEX = 6,
    FPT_STATUS_BAD_CRC = 7,
    FPT_STATUS_BUSY = 8,
    FPT_STATUS_RESET_TIMEOUT = 9,
    FPT_STATUS_RUN_TIMEOUT = 10,
    FPT_STATUS_FPGA_ERROR = 11,
    FPT_STATUS_INTERNAL_ERROR = 12
} fpt_status_code_t;

static inline uint16_t fpt_read_le16(const uint8_t *data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8U);
}

static inline uint32_t fpt_read_le32(const uint8_t *data)
{
    return (uint32_t)data[0]
        | ((uint32_t)data[1] << 8U)
        | ((uint32_t)data[2] << 16U)
        | ((uint32_t)data[3] << 24U);
}

static inline void fpt_write_le16(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)(value & 0xffU);
    data[1] = (uint8_t)((value >> 8U) & 0xffU);
}

static inline void fpt_write_le32(uint8_t *data, uint32_t value)
{
    data[0] = (uint8_t)(value & 0xffU);
    data[1] = (uint8_t)((value >> 8U) & 0xffU);
    data[2] = (uint8_t)((value >> 16U) & 0xffU);
    data[3] = (uint8_t)((value >> 24U) & 0xffU);
}

static inline int fpt_magic_matches(const uint8_t *data)
{
    return data[0] == FPT_MAGIC_BYTE_0
        && data[1] == FPT_MAGIC_BYTE_1
        && data[2] == FPT_MAGIC_BYTE_2
        && data[3] == FPT_MAGIC_BYTE_3;
}

static inline void fpt_write_magic(uint8_t *data)
{
    data[0] = FPT_MAGIC_BYTE_0;
    data[1] = FPT_MAGIC_BYTE_1;
    data[2] = FPT_MAGIC_BYTE_2;
    data[3] = FPT_MAGIC_BYTE_3;
}

#endif /* FPT_ATTENTION_PROTOCOL_H */

