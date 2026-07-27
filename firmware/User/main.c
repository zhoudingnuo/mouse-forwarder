/**
 * @file    main.c
 * @brief   CH32V305 鼠标转发器 - 主程序入口
 * 
 * 功能:
 *   - USB1 (FS): CDC 虚拟串口, 接收 PC 发来的鼠标数据
 *   - USB2 (HS): HID 鼠标设备, 输出到目标电脑
 * 
 * 硬件: nanoCH32V305 (CH32V305RBT6) - 128KB Flash, 32KB RAM
 *   系统时钟: 96MHz (HSE 8MHz * PLL 12x)
 *   USB1 FS: 48MHz (PLL/2)
 */

#include "debug.h"
#include "protocol.h"
#include "mouse_forwarder.h"

/* 全局变量 - USB1 接收缓冲区 */
volatile uint8_t  g_usb1_rx_ready = 0;
volatile uint8_t  g_usb1_rx_buffer[64];
volatile uint16_t g_usb1_rx_len = 0;

/*********************************************************************
 * @fn      main
 * @brief   主程序
 * @return  none
 */
int main(void)
{
    SystemCoreClockUpdate();
    Delay_Init();
    USART_Printf_Init(115200);

    printf("========================================\r\n");
    printf("  CH32V305 Mouse Forwarder v1.0\r\n");
    printf("  USB1: CDC Serial (to Control PC)\r\n");
    printf("  USB2: HID Mouse  (to Target PC)\r\n");
    printf("========================================\r\n");

    /* 初始化 USB1 FS - CDC 虚拟串口 */
    printf("Initializing USB1 (FS CDC)...\r\n");
    USB1_CDC_Init();
    printf("USB1 CDC Ready\r\n");

    /* 初始化 USB2 HS - HID 鼠标 */
    printf("Initializing USB2 (HS HID)...\r\n");
    USB2_HID_Init();
    printf("USB2 HID Mouse Ready\r\n");

    /* 初始化鼠标转发器 */
    MouseForwarder_Init();

    printf("System Ready! Waiting for data...\r\n");

    while (1)
    {
        if (g_usb1_rx_ready)
        {
            g_usb1_rx_ready = 0;
            MouseForwarder_ProcessPacket(
                (uint8_t *)g_usb1_rx_buffer,
                g_usb1_rx_len
            );
        }
        MouseForwarder_Heartbeat();
    }
}