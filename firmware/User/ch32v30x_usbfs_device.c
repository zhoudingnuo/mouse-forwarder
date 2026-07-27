/**
 * @file    ch32v30x_usbfs_device.c
 * @brief   USB1 FS - CDC 虚拟串口
 * 
 * 基于新版 SDK 寄存器定义
 * 
 * 端点:
 *   EP0: Control (64 bytes)
 *   EP1: Bulk IN  (64 bytes) - CDC 数据发送
 *   EP2: Bulk OUT (64 bytes) - CDC 数据接收
 *   EP3: Interrupt IN (8 bytes) - CDC 通知
 */

#include "debug.h"
#include "mouse_forwarder.h"
#include "protocol.h"
#include "ch32v30x_usb_compat.h"

#define USBD_VID            0x4348
#define USBD_PID            0x55E1

/* 端点缓冲区 (需 4 字节对齐) */
__attribute__((aligned(4))) static uint8_t s_ep0_buf[64];
__attribute__((aligned(4))) static uint8_t s_ep1_buf[64];
__attribute__((aligned(4))) static uint8_t s_ep2_buf[64];
__attribute__((aligned(4))) static uint8_t s_ep3_buf[8];

/* CDC 行编码 */
typedef struct {
    uint32_t dwDTERate;
    uint8_t  bCharFormat;
    uint8_t  bParityType;
    uint8_t  bDataBits;
} CDC_LineCoding_t;

static CDC_LineCoding_t s_line = { 115200, 0, 0, 8 };
static volatile uint8_t s_connected = 0;

/* USB 描述符 */
static const uint8_t s_device_desc[] = {
    0x12, 0x01, WBVAL(0x0110), 0x02, 0x00, 0x00, 0x40,
    WBVAL(USBD_VID), WBVAL(USBD_PID), WBVAL(0x0200),
    0x01, 0x02, 0x00, 0x01
};

static const uint8_t s_config_desc[] = {
    /* 配置 */
    0x09, 0x02, WBVAL(67), 0x02, 0x01, 0x00, 0x80, 0x32,
    /* 接口 0: CDC 通信 */
    0x09, 0x04, 0x00, 0x00, 0x01, 0x02, 0x02, 0x01, 0x00,
    0x05, 0x24, 0x00, 0x10, 0x01,
    0x05, 0x24, 0x01, 0x00, 0x01,
    0x04, 0x24, 0x02, 0x02,
    0x05, 0x24, 0x06, 0x00, 0x01,
    /* EP3 IN (通知) */
    0x07, 0x05, 0x83, 0x03, WBVAL(8), 0x10,
    /* 接口 1: CDC 数据 */
    0x09, 0x04, 0x01, 0x00, 0x02, 0x0A, 0x00, 0x00, 0x00,
    /* EP1 IN (Bulk) */
    0x07, 0x05, 0x81, 0x02, WBVAL(64), 0x00,
    /* EP2 OUT (Bulk) */
    0x07, 0x05, 0x02, 0x02, WBVAL(64), 0x00,
};

static const uint8_t s_string_lang[]    = { 0x04, 0x03, 0x09, 0x04 };
static const uint8_t s_string_vendor[]  = { 0x0E, 0x03, 'W',0,'C',0,'H',0,' ',0,'C',0,'o',0,'.',0 };
static const uint8_t s_string_product[] = { 0x24, 0x03, 'C',0,'H',0,'3',0,'2',0,'V',0,'3',0,'0',0,'5',0,' ',0,'M',0,'o',0,'u',0,'s',0,'e',0,' ',0,'F',0,'w',0,'d',0,'r',0 };

void USB1_CDC_Init(void)
{
    /* 时钟: 48MHz */
    RCC_USBFSCLKConfig(RCC_USBFSCLKSource_PLLCLK_Div2);
    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_USBFS, ENABLE);
    
    /* 使能上拉 */
    USBFSD->BASE_CTRL = 0x01;
    USBFSD->DEV_ADDR = 0;
    
    /* EP0: 64 字节, DMA */
    USBFSD->UEP0_DMA = (uint32_t)s_ep0_buf;
    USBFSD->UEP0_TX_LEN = 0;
    USBFSD->UEP0_TX_CTRL = 0x12;  /* TX: NAK */
    USBFSD->UEP0_RX_CTRL = 0x01;  /* RX: ACK */
    
    /* EP1: Bulk IN, 64 字节 */
    USBFSD->UEP1_DMA = (uint32_t)s_ep1_buf;
    USBFSD->UEP1_TX_CTRL = 0x12;  /* NAK */
    
    /* EP2: Bulk OUT, 64 字节 */
    USBFSD->UEP2_DMA = (uint32_t)s_ep2_buf;
    USBFSD->UEP2_RX_CTRL = 0x01;  /* ACK */
    
    /* EP3: Interrupt IN, 8 字节 */
    USBFSD->UEP3_DMA = (uint32_t)s_ep3_buf;
    USBFSD->UEP3_TX_CTRL = 0x12;  /* NAK */
    
    /* 端点模式: EP1=批量, EP2=批量, EP3=中断 */
    USBFSD->UEP2_3_MOD = 0x42;  /* EP2=批量, EP3=中断 */
    
    /* 中断 */
    USBFSD->INT_EN = 0x02;  /* 传输完成中断 */
    NVIC_EnableIRQ(USBFS_IRQn);
    
    printf("USB1 CDC: VID=%04X PID=%04X\r\n", USBD_VID, USBD_PID);
}

void USB1_CDC_SendData(uint8_t *data, uint16_t len)
{
    uint16_t send_len = (len > 64) ? 64 : len;
    for (uint16_t i = 0; i < send_len; i++) s_ep1_buf[i] = data[i];
    USBFSD->UEP1_TX_LEN = send_len;
    USBFSD->UEP1_TX_CTRL = 0x00;  /* ACK */
}

void USBFS_IRQHandler(void) __attribute__((interrupt("WCH-Interrupt-fast")));
void USBFS_IRQHandler(void)
{
    uint8_t intfg = USBFSD->INT_FG;
    uint8_t intst = USBFSD->INT_ST;
    uint8_t *buf = s_ep0_buf;
    
    if (intfg & 0x02)  /* 传输完成 */
    {
        uint8_t token = intst & 0x60;
        
        if (token == 0x00)  /* SETUP */
        {
            uint8_t req = buf[1];
            uint8_t type = buf[0];
            
            if ((type & 0x60) == 0x00)  /* 标准请求 */
            {
                switch (req)
                {
                case 0x05:  /* SET_ADDRESS */
                    USBFSD->DEV_ADDR = buf[2];
                    USBFSD->UEP0_TX_CTRL = 0x00;
                    USBFSD->UEP0_RX_CTRL = 0x01;
                    break;
                    
                case 0x06:  /* GET_DESCRIPTOR */
                {
                    uint8_t dt = buf[3], di = buf[2];
                    const uint8_t *d = NULL;
                    uint16_t dl = 0;
                    switch (dt) {
                        case 0x01: d = s_device_desc; dl = sizeof(s_device_desc); break;
                        case 0x02: d = s_config_desc; dl = sizeof(s_config_desc); break;
                        case 0x03:
                            if (di == 0) d = s_string_lang;
                            else if (di == 1) d = s_string_vendor;
                            else if (di == 2) d = s_string_product;
                            if (d) dl = d[0];
                            break;
                    }
                    if (d) {
                        uint16_t rl = (buf[7] << 8) | buf[6];
                        uint16_t cl = (dl < rl) ? dl : rl;
                        for (uint16_t i = 0; i < cl; i++) s_ep0_buf[i] = d[i];
                        USBFSD->UEP0_TX_LEN = cl;
                    }
                    USBFSD->UEP0_TX_CTRL = 0x00;
                    USBFSD->UEP0_RX_CTRL = 0x01;
                    break;
                }
                
                case 0x09:  /* SET_CONFIGURATION */
                    s_connected = 1;
                    USBFSD->UEP0_TX_CTRL = 0x00;
                    USBFSD->UEP0_RX_CTRL = 0x01;
                    printf("USB1 CDC: Configured\r\n");
                    break;
                    
                default:
                    USBFSD->UEP0_TX_CTRL = 0x00;
                    USBFSD->UEP0_RX_CTRL = 0x01;
                    break;
                }
            }
            else if ((type & 0x60) == 0x20)  /* 类请求 */
            {
                switch (req)
                {
                case 0x20:  /* SET_LINE_CODING */
                    s_line.dwDTERate = buf[2] | (buf[3]<<8) | (buf[4]<<16) | (buf[5]<<24);
                    s_line.bCharFormat = buf[6];
                    s_line.bParityType = buf[7];
                    s_line.bDataBits = buf[8];
                    printf("CDC: %lu bps\r\n", (unsigned long)s_line.dwDTERate);
                    break;
                case 0x21:  /* GET_LINE_CODING */
                    buf[0] = (uint8_t)s_line.dwDTERate;
                    buf[1] = (uint8_t)(s_line.dwDTERate >> 8);
                    buf[2] = (uint8_t)(s_line.dwDTERate >> 16);
                    buf[3] = (uint8_t)(s_line.dwDTERate >> 24);
                    buf[4] = s_line.bCharFormat;
                    buf[5] = s_line.bParityType;
                    buf[6] = s_line.bDataBits;
                    USBFSD->UEP0_TX_LEN = 7;
                    break;
                case 0x22:  /* SET_CONTROL_LINE_STATE */
                    break;
                }
                USBFSD->UEP0_TX_CTRL = 0x00;
                USBFSD->UEP0_RX_CTRL = 0x01;
            }
            else
            {
                USBFSD->UEP0_TX_CTRL = 0x00;
                USBFSD->UEP0_RX_CTRL = 0x01;
            }
        }
        else if (token == 0x40)  /* IN */
        {
            USBFSD->UEP0_TX_CTRL = 0x00;
            USBFSD->UEP0_RX_CTRL = 0x01;
        }
        else if (token == 0x20)  /* OUT */
        {
            uint8_t ep = intst & 0x0F;
            if (ep == 2) {
                uint16_t rx_len = USBFSD->RX_LEN & 0x3FF;
                for (uint16_t i = 0; i < rx_len && i < 64; i++)
                    g_usb1_rx_buffer[i] = s_ep2_buf[i];
                g_usb1_rx_len = (rx_len > 64) ? 64 : rx_len;
                g_usb1_rx_ready = 1;
                USBFSD->UEP2_RX_CTRL = 0x01;
            }
            USBFSD->UEP0_TX_CTRL = 0x00;
            USBFSD->UEP0_RX_CTRL = 0x01;
        }
        
        USBFSD->INT_FG = 0x02;
    }
}