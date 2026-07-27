/**
 * @file    main.c
 * @brief   CH32V305 鼠标转发器 - 双 USB 口
 * 
 * USB1 (FS): 纯 CDC 虚拟串口 - 接收本机鼠标数据
 * USB2 (HS): HID 鼠标 - 输出到目标电脑
 * 
 * 通信协议 (6 字节):
 *   [0xAA] [flags] [dx] [dy] [wheel] [checksum]
 */

#include "UART.h"
#include "debug.h"
#include "ch32v30x_usbfs_device.h"
#include "ch32v30x_usbhs_device.h"

extern volatile uint8_t USBFS_DevEnumStatus;
extern volatile uint8_t USBHS_DevEnumStatus;

/* 鼠标报告缓冲区 */
__attribute__((aligned(4))) uint8_t Mouse_Report[4] = {0, 0, 0, 0};

/* 协议解析状态 */
volatile uint8_t  Rx_Packet[6];
volatile uint8_t  Rx_Idx = 0;
volatile uint8_t  Rx_Packet_Ready = 0;

#define PKG_HEADER   0xAA
#define PKG_SIZE     6
#define FLAG_LEFT    0x01
#define FLAG_RIGHT   0x02
#define FLAG_MIDDLE  0x04
#define FLAG_WHEEL   0x08

void Mouse_Packet_Parse(uint8_t data)
{
    if(Rx_Idx == 0)
    {
        if(data == PKG_HEADER)
            Rx_Packet[Rx_Idx++] = data;
    }
    else if(Rx_Idx < PKG_SIZE)
    {
        Rx_Packet[Rx_Idx++] = data;
        if(Rx_Idx >= PKG_SIZE)
        {
            uint8_t sum = 0, i;
            for(i = 0; i < PKG_SIZE - 1; i++)
                sum ^= Rx_Packet[i];
            if(sum == Rx_Packet[PKG_SIZE - 1])
                Rx_Packet_Ready = 1;
            Rx_Idx = 0;
        }
    }
    else
        Rx_Idx = 0;
}

void Mouse_Send_HID_Report(void)
{
    if(!USBHS_DevEnumStatus)
        return;

    /* 如果有新数据, 更新鼠标报告 */
    if(Rx_Packet_Ready)
    {
        uint8_t flags = Rx_Packet[1];
        Mouse_Report[0] = flags & 0x07;
        Mouse_Report[1] = Rx_Packet[2];
        Mouse_Report[2] = Rx_Packet[3];
        Mouse_Report[3] = (flags & FLAG_WHEEL) ? Rx_Packet[4] : 0;
        Rx_Packet_Ready = 0;
    }
    /* 不管有没有新数据, 都用最后的状态发送 HID 报告 */
    USBHS_Endp_DataUp(DEF_UEP2, Mouse_Report, 4, DEF_UEP_CPY_LOAD);
}

int main(void)
{
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
    SystemCoreClockUpdate();
    Delay_Init();
    USART_Printf_Init(115200);

    printf("========================================\r\n");
    printf("  CH32V305 Mouse Forwarder v2.0\r\n");
    printf("  USB1 (FS): CDC Serial - receive\r\n");
    printf("  USB2 (HS): HID Mouse - output\r\n");
    printf("========================================\r\n");

    /* 初始化 USB1 FS - 纯 CDC 虚拟串口 */
    RCC_Configuration();
    TIM2_Init();
    UART2_Init(1, DEF_UARTx_BAUDRATE, DEF_UARTx_STOPBIT, DEF_UARTx_PARITY);
    USBFS_RCC_Init();
    USBFS_Device_Init(ENABLE);
    printf("USB1 CDC Ready\r\n");

    /* 初始化 USB2 HS - HID 鼠标 */
    USBHS_RCC_Init();
    USBHS_Device_Init(ENABLE);
    printf("USB2 HID Mouse Ready\r\n");

    printf("System Ready!\r\n");

    while(1)
    {
        UART2_DataRx_Deal();
        UART2_DataTx_Deal();
        Mouse_Send_HID_Report();
    }
}