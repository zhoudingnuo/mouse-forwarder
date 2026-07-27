/**
 * @file    main.c
 * @brief   CH32V305 鼠标转发器 - CDC + HID 鼠标复合设备
 * 
 * 基于 WCH 官方 SimulateCDC-HID 示例
 * CDC 虚拟串口接收鼠标数据 → HID 鼠标输出
 * 
 * 通信协议 (6 字节):
 *   [0xAA] [flags] [dx] [dy] [wheel] [checksum]
 */

#include "UART.h"
#include "debug.h"
#include "ch32v30x_usbfs_device.h"

/* 鼠标报告缓冲区 (4字节: buttons, X, Y, wheel) */
__attribute__((aligned(4))) uint8_t Mouse_Report[4] = {0, 0, 0, 0};

/* 协议解析状态 */
volatile uint8_t  Rx_Packet[6];
volatile uint8_t  Rx_Idx = 0;
volatile uint8_t  Rx_Packet_Ready = 0;

/* 统计 */
volatile uint32_t Pkt_OK = 0;
volatile uint32_t Pkt_Err = 0;

/* 协议常量 */
#define PKG_HEADER   0xAA
#define PKG_SIZE     6
#define FLAG_LEFT    0x01
#define FLAG_RIGHT   0x02
#define FLAG_MIDDLE  0x04
#define FLAG_WHEEL   0x08

/*********************************************************************
 * @fn      Mouse_Packet_Parse
 * @brief   解析收到的字节, 组装成鼠标协议包
 * @param   data - 收到的字节
 */
void Mouse_Packet_Parse(uint8_t data)
{
    if(Rx_Idx == 0)
    {
        if(data == PKG_HEADER)
        {
            Rx_Packet[Rx_Idx++] = data;
        }
    }
    else if(Rx_Idx < PKG_SIZE)
    {
        Rx_Packet[Rx_Idx++] = data;
        if(Rx_Idx >= PKG_SIZE)
        {
            /* 校验 */
            uint8_t sum = 0;
            uint8_t i;
            for(i = 0; i < PKG_SIZE - 1; i++)
                sum ^= Rx_Packet[i];

            if(sum == Rx_Packet[PKG_SIZE - 1])
            {
                Rx_Packet_Ready = 1;
                Pkt_OK++;
            }
            else
            {
                Pkt_Err++;
            }
            Rx_Idx = 0;
        }
    }
    else
    {
        Rx_Idx = 0;
    }
}

/*********************************************************************
 * @fn      Mouse_Send_HID_Report
 * @brief   通过 HID 端点发送鼠标报告
 */
void Mouse_Send_HID_Report(void)
{
    if(!Rx_Packet_Ready)
        return;

    /* 检查 EP4 是否空闲 (HID 接口用 EP4) */
    if(USBFS_Endp_Busy[DEF_UEP4] == 0)
    {
        /* 构造 HID 鼠标报告 */
        uint8_t flags = Rx_Packet[1];
        Mouse_Report[0] = flags & 0x07;  /* 按钮状态 */
        Mouse_Report[1] = Rx_Packet[2];  /* X 位移 */
        Mouse_Report[2] = Rx_Packet[3];  /* Y 位移 */
        Mouse_Report[3] = (flags & FLAG_WHEEL) ? Rx_Packet[4] : 0;  /* 滚轮 */

        /* 通过 EP4 发送 (HID 接口用的是 EP4) */
        USBFS_Endp_DataUp(DEF_UEP4, Mouse_Report, 4, DEF_UEP_CPY_LOAD);
        Rx_Packet_Ready = 0;
    }
}

/*********************************************************************
 * @fn      main
 */
int main(void)
{
    SystemCoreClockUpdate();
    Delay_Init();
    USART_Printf_Init(115200);

    printf("========================================\r\n");
    printf("  CH32V305 Mouse Forwarder v1.0\r\n");
    printf("  CDC: Receive mouse data from PC\r\n");
    printf("  HID: Mouse output to target PC\r\n");
    printf("========================================\r\n");
    printf("SystemClk: %d\r\n", SystemCoreClock);
    printf("ChipID: %08x\r\n", DBGMCU_GetCHIPID());

    RCC_Configuration();
    TIM2_Init();

    /* UART2 init (保留, 但不使用 USART2 转发) */
    UART2_Init(1, DEF_UARTx_BAUDRATE, DEF_UARTx_STOPBIT, DEF_UARTx_PARITY);

    /* USB device init */
    USBFS_RCC_Init();
    USBFS_Device_Init(ENABLE);

    printf("USB CDC+HID Mouse Ready\r\n");

    while(1)
    {
        /* 处理 UART2 数据接收 (从 USART2 收到的数据会触发 USB 上传) */
        UART2_DataRx_Deal();
        UART2_DataTx_Deal();

        /* 发送 HID 鼠标报告 */
        Mouse_Send_HID_Report();
    }
}