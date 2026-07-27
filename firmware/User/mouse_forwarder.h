/**
 * @file    mouse_forwarder.h
 * @brief   鼠标转发器核心逻辑
 */

#ifndef __MOUSE_FORWARDER_H
#define __MOUSE_FORWARDER_H

#include <stdint.h>
#include "protocol.h"

/* 心跳间隔 (ms) */
#define HEARTBEAT_INTERVAL_MS  100

/* 外部全局变量声明 */
extern volatile uint8_t  g_usb1_rx_ready;
extern volatile uint8_t  g_usb1_rx_buffer[64];
extern volatile uint16_t g_usb1_rx_len;

/* USB1 FS CDC 初始化 */
void USB1_CDC_Init(void);

/* USB2 HS HID 初始化 */
void USB2_HID_Init(void);

/* 鼠标转发器初始化 */
void MouseForwarder_Init(void);

/* 处理收到的数据包 */
void MouseForwarder_ProcessPacket(uint8_t *data, uint16_t len);

/* 心跳维护 */
void MouseForwarder_Heartbeat(void);

/* 通过 USB2 HS 发送 HID 鼠标报告 */
void USB2_SendMouseReport(MouseReport_t *report);

/* 通过 USB1 FS 发送 CDC 数据 (调试/回显) */
void USB1_CDC_SendData(uint8_t *data, uint16_t len);

#endif /* __MOUSE_FORWARDER_H */