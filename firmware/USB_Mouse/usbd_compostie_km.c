/**
 * @file    usbd_compostie_km.c
 * @brief   USB HID 鼠标处理 - 接收 USART2 数据并转发为 HID 报告
 * 
 * 基于 WCH 官方 CompositeKM 示例，简化为只保留鼠标部分
 * 
 * 通信协议 (6 字节):
 *   [0xAA] [flags] [dx] [dy] [wheel] [checksum]
 *   flags: bit0=左键, bit1=右键, bit2=中键, bit3=滚轮有效
 */

#include "ch32v30x_usbfs_device.h"
#include "usbd_composite_km.h"

/* 协议定义 */
#define PKG_HEADER       0xAA
#define PKG_SIZE         6
#define FLAG_LEFT        0x01
#define FLAG_RIGHT       0x02
#define FLAG_MIDDLE      0x04
#define FLAG_WHEEL       0x08

/* 鼠标报告缓冲区 (4字节: buttons, X, Y, wheel) */
uint8_t  MS_Data_Pack[ 4 ] = { 0x00 };

/* 键盘 LED 状态 (被 USB 中断引用) */
volatile uint8_t  KB_LED_Last_Status = 0x00;
volatile uint8_t  KB_LED_Cur_Status = 0x00;

/* USART 接收缓冲 */
volatile uint8_t  USART_Rx_Buf[ 16 ];
volatile uint8_t  USART_Rx_Idx = 0;
volatile uint8_t  USART_Packet_Ready = 0;

/* 统计信息 */
volatile uint32_t Pkt_Received = 0;
volatile uint32_t Pkt_Invalid = 0;

/* 中断声明 */
void USART2_IRQHandler(void) __attribute__((interrupt("WCH-Interrupt-fast")));

/*********************************************************************
 * @fn      USART2_Init
 * @brief   初始化 USART2 (PA2=TX, PA3=RX)
 * @param   baudrate - 波特率
 */
void USART2_Init( uint32_t baudrate )
{
    GPIO_InitTypeDef  GPIO_InitStructure = {0};
    USART_InitTypeDef USART_InitStructure = {0};
    NVIC_InitTypeDef  NVIC_InitStructure = {0};

    RCC_APB2PeriphClockCmd( RCC_APB2Periph_GPIOA, ENABLE );
    RCC_APB1PeriphClockCmd( RCC_APB1Periph_USART2, ENABLE );

    /* PA2 (USART2_TX) - 复用推挽输出 */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init( GPIOA, &GPIO_InitStructure );

    /* PA3 (USART2_RX) - 浮空输入 */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_3;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init( GPIOA, &GPIO_InitStructure );

    /* USART2 配置 */
    USART_InitStructure.USART_BaudRate = baudrate;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Tx | USART_Mode_Rx;
    USART_Init( USART2, &USART_InitStructure );

    /* 使能接收中断 */
    USART_ITConfig( USART2, USART_IT_RXNE, ENABLE );

    /* NVIC 配置 */
    NVIC_InitStructure.NVIC_IRQChannel = USART2_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init( &NVIC_InitStructure );

    USART_Cmd( USART2, ENABLE );
}

/*********************************************************************
 * @fn      USART2_IRQHandler
 * @brief   USART2 接收中断 - 解析协议包
 */
void USART2_IRQHandler( void )
{
    if( USART_GetITStatus( USART2, USART_IT_RXNE ) != RESET )
    {
        uint8_t data = USART_ReceiveData( USART2 ) & 0xFF;

        /* 状态机解析协议包 */
        if( USART_Rx_Idx == 0 )
        {
            /* 等待帧头 0xAA */
            if( data == PKG_HEADER )
            {
                USART_Rx_Buf[ USART_Rx_Idx++ ] = data;
            }
        }
        else if( USART_Rx_Idx < PKG_SIZE )
        {
            /* 接收剩余字节 */
            USART_Rx_Buf[ USART_Rx_Idx++ ] = data;

            /* 接收完整包 */
            if( USART_Rx_Idx >= PKG_SIZE )
            {
                /* 验证校验和 */
                uint8_t checksum = 0;
                uint8_t i;
                for( i = 0; i < PKG_SIZE - 1; i++ )
                {
                    checksum ^= USART_Rx_Buf[ i ];
                }

                if( checksum == USART_Rx_Buf[ PKG_SIZE - 1 ] )
                {
                    /* 有效数据包 */
                    USART_Packet_Ready = 1;
                    Pkt_Received++;
                }
                else
                {
                    Pkt_Invalid++;
                }

                USART_Rx_Idx = 0;
            }
        }
        else
        {
            USART_Rx_Idx = 0;
        }
    }
}

/*********************************************************************
 * @fn      Mouse_Data_Handle
 * @brief   处理接收到的鼠标数据, 转发为 HID 报告
 */
void Mouse_Data_Handle( void )
{
    uint8_t status;

    if( USART_Packet_Ready )
    {
        USART_Packet_Ready = 0;

        /* 从协议包提取鼠标数据 */
        uint8_t flags = USART_Rx_Buf[ 1 ];
        int8_t  dx    = (int8_t)USART_Rx_Buf[ 2 ];
        int8_t  dy    = (int8_t)USART_Rx_Buf[ 3 ];
        int8_t  wheel = ( flags & FLAG_WHEEL ) ? (int8_t)USART_Rx_Buf[ 4 ] : 0;

        /* 构造 HID 鼠标报告 */
        MS_Data_Pack[ 0 ] = flags & 0x07;  /* 按钮状态 */
        MS_Data_Pack[ 1 ] = (uint8_t)dx;   /* X 位移 */
        MS_Data_Pack[ 2 ] = (uint8_t)dy;   /* Y 位移 */
        MS_Data_Pack[ 3 ] = (uint8_t)wheel;/* 滚轮 */

        /* 通过 USBFS EP2 发送 HID 鼠标报告 */
        status = USBFS_Endp_DataUp( DEF_UEP2, MS_Data_Pack, sizeof( MS_Data_Pack ), DEF_UEP_CPY_LOAD );

        if( status != READY )
        {
            /* 端点忙, 重新标记为待处理 */
            USART_Packet_Ready = 1;
        }
    }
}

/*********************************************************************
 * @fn      USB_Sleep_Wakeup_CFG
 * @brief   配置 USB 唤醒 (保留接口, 空实现)
 */
void USB_Sleep_Wakeup_CFG( void )
{
}

/*********************************************************************
 * @fn      MCU_Sleep_Wakeup_Operate
 * @brief   休眠唤醒操作 (保留接口, 空实现)
 */
void MCU_Sleep_Wakeup_Operate( void )
{
}