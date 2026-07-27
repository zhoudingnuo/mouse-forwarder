/**
 * @file    ch32v30x_usbhs_device.c
 * @brief   USB2 HS - HID 鼠标设备实现
 * 
 * 基于新版 SDK 寄存器定义
 * 
 * 端点配置:
 *   EP0: Control (64 bytes)
 *   EP1: Interrupt IN (64 bytes) - HID 鼠标报告
 */

#include "debug.h"
#include "mouse_forwarder.h"
#include "protocol.h"
#include "ch32v30x_usb_compat.h"

#define USBD_VID            0x4348
#define USBD_PID            0x55E2

/* HID 报告描述符 */
static const uint8_t s_hid_report_desc[] = {
    0x05, 0x01,             /* Usage Page (Generic Desktop) */
    0x09, 0x02,             /* Usage (Mouse) */
    0xA1, 0x01,             /* Collection (Application) */
    0x09, 0x01,             /*   Usage (Pointer) */
    0xA1, 0x00,             /*   Collection (Physical) */
    0x05, 0x09,             /*     Usage Page (Button) */
    0x19, 0x01,             /*     Usage Minimum (Button 1) */
    0x29, 0x03,             /*     Usage Maximum (Button 3) */
    0x15, 0x00,             /*     Logical Minimum (0) */
    0x25, 0x01,             /*     Logical Maximum (1) */
    0x95, 0x03,             /*     Report Count (3) */
    0x75, 0x01,             /*     Report Size (1) */
    0x81, 0x02,             /*     Input (Data,Var,Abs) */
    0x95, 0x01,             /*     Report Count (1) */
    0x75, 0x05,             /*     Report Size (5) */
    0x81, 0x03,             /*     Input (Const,Var,Abs) */
    0x05, 0x01,             /*     Usage Page (Generic Desktop) */
    0x09, 0x30,             /*     Usage (X) */
    0x09, 0x31,             /*     Usage (Y) */
    0x09, 0x38,             /*     Usage (Wheel) */
    0x15, 0x81,             /*     Logical Minimum (-127) */
    0x25, 0x7F,             /*     Logical Maximum (127) */
    0x75, 0x08,             /*     Report Size (8) */
    0x95, 0x03,             /*     Report Count (3) */
    0x81, 0x06,             /*     Input (Data,Var,Rel) */
    0xC0,                   /*   End Collection */
    0xC0                    /* End Collection */
};

/* USB 描述符 */
static const uint8_t s_device_desc[] = {
    0x12, 0x01, WBVAL(0x0200), 0x00, 0x00, 0x00, 0x40,
    WBVAL(USBD_VID), WBVAL(USBD_PID), WBVAL(0x0200),
    0x01, 0x02, 0x00, 0x01
};

static const uint8_t s_config_desc[] = {
    0x09, 0x02, WBVAL(41), 0x01, 0x01, 0x00, 0xA0, 0x32,
    0x09, 0x04, 0x00, 0x00, 0x01, 0x03, 0x01, 0x02, 0x00,
    0x09, 0x21, WBVAL(0x0111), 0x00, 0x01, 0x22, WBVAL(sizeof(s_hid_report_desc)),
    0x07, 0x05, 0x81, 0x03, WBVAL(64), 0x01
};

static const uint8_t s_string_lang[]    = { 0x04, 0x03, 0x09, 0x04 };
static const uint8_t s_string_vendor[]  = { 0x0E, 0x03, 'W',0,'C',0,'H',0,' ',0,'C',0,'o',0,'.',0 };
static const uint8_t s_string_product[] = { 0x1E, 0x03, 'C',0,'H',0,'3',0,'2',0,'V',0,'3',0,'0',0,'5',0,' ',0,'H',0,'I',0,'D',0,' ',0,'M',0,'o',0,'u',0,'s',0,'e',0 };

static volatile uint8_t s_configured = 0;
static MouseReport_t s_report = {0, 0, 0, 0};

/* 端点缓冲区 (需 4 字节对齐) */
__attribute__((aligned(4))) static uint8_t s_ep0_buf[64];
__attribute__((aligned(4))) static uint8_t s_ep1_tx_buf[64];

void USB2_HID_Init(void)
{
    /* 使能时钟 */
    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_USBHS, ENABLE);
    
    /* 配置 USBHS PLL: HSE=8MHz → 48MHz */
    RCC_USBHSPLLCLKConfig(RCC_HSBHSPLLCLKSource_HSE);
    RCC_USBHSPLLCKREFCLKConfig(RCC_USBHSPLLCKREFCLK_8M);
    
    /* 复位 */
    USBHSD->HOST_CTRL = 0x01;
    Delay_Us(10000);
    USBHSD->HOST_CTRL = 0;
    
    /* 配置设备模式 */
    USBHSD->CONTROL = 0x01;  /* SIE 使能 */
    Delay_Us(1000);
    
    /* 配置端点 0: 64 字节, DMA 地址 */
    USBHSD->UEP0_DMA = (uint32_t)s_ep0_buf;
    USBHSD->UEP0_MAX_LEN = 64;
    USBHSD->UEP0_TX_CTRL = 0x12;  /* NAK */
    USBHSD->UEP0_RX_CTRL = 0x01;  /* ACK */
    
    /* 配置端点 1: 64 字节, IN, 中断传输 */
    USBHSD->UEP1_TX_DMA = (uint32_t)s_ep1_tx_buf;
    USBHSD->UEP1_MAX_LEN = 64;
    USBHSD->UEP1_TX_CTRL = 0x12;  /* NAK */
    
    /* 使能端点中断 */
    USBHSD->INT_EN = 0x03;  /* EP0 + EP1 */
    
    NVIC_EnableIRQ(USBHS_IRQn);
    printf("USB2 HID: VID=%04X PID=%04X\r\n", USBD_VID, USBD_PID);
}

void USB2_SendMouseReport(MouseReport_t *report)
{
    if (!s_configured) return;
    
    s_report.buttons = report->buttons;
    s_report.x = report->x;
    s_report.y = report->y;
    s_report.wheel = report->wheel;
    
    s_ep1_tx_buf[0] = s_report.buttons;
    s_ep1_tx_buf[1] = (uint8_t)s_report.x;
    s_ep1_tx_buf[2] = (uint8_t)s_report.y;
    s_ep1_tx_buf[3] = (uint8_t)s_report.wheel;
    
    USBHSD->UEP1_TX_LEN = 4;
    USBHSD->UEP1_TX_CTRL = 0x00;  /* ACK */
}

void USBHS_IRQHandler(void) __attribute__((interrupt("WCH-Interrupt-fast")));
void USBHS_IRQHandler(void)
{
    uint8_t intfg = USBHSD->INT_FG;
    uint8_t intst = USBHSD->INT_ST;
    uint8_t *buf = s_ep0_buf;
    
    /* 端点 0 中断 */
    if (intfg & 0x01)
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
                    USBHSD->DEV_AD = buf[2];
                    USBHSD->UEP0_TX_CTRL = 0x00;
                    USBHSD->UEP0_RX_CTRL = 0x01;
                    break;
                    
                case 0x06:  /* GET_DESCRIPTOR */
                {
                    uint8_t desc_type = buf[3];
                    uint8_t desc_idx = buf[2];
                    const uint8_t *desc = NULL;
                    uint16_t desc_len = 0;
                    
                    switch (desc_type)
                    {
                    case 0x01: desc = s_device_desc; desc_len = sizeof(s_device_desc); break;
                    case 0x02: desc = s_config_desc; desc_len = sizeof(s_config_desc); break;
                    case 0x22: desc = s_hid_report_desc; desc_len = sizeof(s_hid_report_desc); break;
                    case 0x03:
                        if (desc_idx == 0) desc = s_string_lang;
                        else if (desc_idx == 1) desc = s_string_vendor;
                        else if (desc_idx == 2) desc = s_string_product;
                        if (desc) desc_len = desc[0];
                        break;
                    }
                    
                    if (desc)
                    {
                        uint16_t req_len = (buf[7] << 8) | buf[6];
                        uint16_t copy_len = (desc_len < req_len) ? desc_len : req_len;
                        for (uint16_t i = 0; i < copy_len; i++) s_ep0_buf[i] = desc[i];
                        USBHSD->UEP0_TX_LEN = copy_len;
                    }
                    USBHSD->UEP0_TX_CTRL = 0x00;
                    USBHSD->UEP0_RX_CTRL = 0x01;
                    break;
                }
                
                case 0x09:  /* SET_CONFIGURATION */
                    s_configured = 1;
                    USBHSD->UEP0_TX_CTRL = 0x00;
                    USBHSD->UEP0_RX_CTRL = 0x01;
                    printf("USB2 HID: Configured\r\n");
                    break;
                    
                default:
                    USBHSD->UEP0_TX_CTRL = 0x00;
                    USBHSD->UEP0_RX_CTRL = 0x01;
                    break;
                }
            }
            else if ((type & 0x60) == 0x20)  /* 类请求 */
            {
                switch (req)
                {
                case 0x01:  /* GET_REPORT */
                    s_ep0_buf[0] = s_report.buttons;
                    s_ep0_buf[1] = (uint8_t)s_report.x;
                    s_ep0_buf[2] = (uint8_t)s_report.y;
                    s_ep0_buf[3] = (uint8_t)s_report.wheel;
                    USBHSD->UEP0_TX_LEN = 4;
                    break;
                }
                USBHSD->UEP0_TX_CTRL = 0x00;
                USBHSD->UEP0_RX_CTRL = 0x01;
            }
            else
            {
                USBHSD->UEP0_TX_CTRL = 0x00;
                USBHSD->UEP0_RX_CTRL = 0x01;
            }
        }
        else if (token == 0x40)  /* IN */
        {
            USBHSD->UEP0_TX_CTRL = 0x00;
            USBHSD->UEP0_RX_CTRL = 0x01;
        }
        else if (token == 0x20)  /* OUT */
        {
            USBHSD->UEP0_TX_CTRL = 0x00;
            USBHSD->UEP0_RX_CTRL = 0x01;
        }
        
        USBHSD->INT_FG = 0x01;
    }
    
    /* 端点 1 中断 (IN 完成) */
    if (intfg & 0x02)
    {
        USBHSD->INT_FG = 0x02;
    }
}