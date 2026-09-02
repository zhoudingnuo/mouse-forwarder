/**
 * @file    protocol.h
 * @brief   PC ↔ CH32V305 通信协议定义
 * 
 * 数据包格式 (6 字节):
 *   Byte 0: Header      0xAA
 *   Byte 1: Flags       bit0=左键, bit1=右键, bit2=中键, bit3=滚轮有效, bit4=后退键, bit5=前进键
 *   Byte 2: Delta X     int8, 相对位移 (-128~127)
 *   Byte 3: Delta Y     int8, 相对位移 (-128~127)
 *   Byte 4: Wheel       int8, 滚轮增量 (-128~127)
 *   Byte 5: Checksum    字节 0~4 的异或和
 */

#ifndef __PROTOCOL_H
#define __PROTOCOL_H

#include <stdint.h>

/* 协议常量 */
#define PKG_HEADER          0xAA
#define PKG_SIZE            6

/* Flags 位定义 */
#define FLAG_LEFT_BUTTON    (1 << 0)
#define FLAG_RIGHT_BUTTON   (1 << 1)
#define FLAG_MIDDLE_BUTTON  (1 << 2)
#define FLAG_WHEEL_VALID    (1 << 6)  /* bit6, 避开 bit3 (HID 后退键位) */
#define FLAG_BACK_BUTTON    (1 << 4)  /* 侧键后退 (X1) */
#define FLAG_FORWARD_BUTTON (1 << 5)  /* 侧键前进 (X2) */

/* 鼠标报告结构体 (4字节, 符合 USB HID 鼠标 Boot Protocol) */
typedef struct __attribute__((packed)) {
    uint8_t buttons;    /* bit0=左键, bit1=右键, bit2=中键, bit4=后退, bit5=前进 */
    int8_t  x;          /* X 轴相对位移 */
    int8_t  y;          /* Y 轴相对位移 */
    int8_t  wheel;      /* 滚轮增量 */
} MouseReport_t;

/* 串口数据包结构体 */
typedef struct __attribute__((packed)) {
    uint8_t  header;    /* 0xAA */
    uint8_t  flags;     /* 按钮和滚轮标志 */
    int8_t   dx;        /* X 相对位移 */
    int8_t   dy;        /* Y 相对位移 */
    int8_t   wheel;     /* 滚轮增量 */
    uint8_t  checksum;  /* 校验和 */
} SerialPacket_t;

/**
 * @brief  计算校验和
 * @param  data  数据指针
 * @param  len   数据长度
 * @return 异或和
 */
static inline uint8_t calc_checksum(const uint8_t *data, uint8_t len)
{
    uint8_t sum = 0;
    for (uint8_t i = 0; i < len; i++) {
        sum ^= data[i];
    }
    return sum;
}

/**
 * @brief  验证数据包
 * @param  pkt  数据包指针
 * @return 1=有效, 0=无效
 */
static inline uint8_t validate_packet(const SerialPacket_t *pkt)
{
    if (pkt->header != PKG_HEADER)
        return 0;
    uint8_t sum = calc_checksum((const uint8_t *)pkt, PKG_SIZE - 1);
    return (sum == pkt->checksum) ? 1 : 0;
}

/**
 * @brief  将串口数据包转换为 HID 鼠标报告
 * @param  pkt   串口数据包
 * @param  report HID 鼠标报告输出
 */
static inline void packet_to_report(const SerialPacket_t *pkt, MouseReport_t *report)
{
    report->buttons = pkt->flags & 0x3F;  /* 保留低 6 位 (左+右+中+后退+前进) */
    report->x       = pkt->dx;
    report->y       = pkt->dy;
    report->wheel   = (pkt->flags & FLAG_WHEEL_VALID) ? pkt->wheel : 0;
}

#endif /* __PROTOCOL_H */