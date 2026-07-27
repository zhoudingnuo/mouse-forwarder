/**
 * ch32v30x_usb_compat.h - USB 寄存器兼容层
 * 将旧版 SDK 寄存器名映射到新版 SDK
 */

#ifndef __USB_COMPAT_H
#define __USB_COMPAT_H

#include "ch32v30x.h"

/* USBFS 映射: 旧版 → 新版 */
#define USBFS               USBFSD
#define USBFS_BASE_CTRL_UDE 0x01

/* UEP 控制寄存器: 旧版合并 → 新版拆分 */
#define USBFS_UEP0_CTRL      (*(__IO uint8_t *)&USBFSD->UEP0_TX_CTRL)
#define USBFS_UEP1_CTRL      (*(__IO uint8_t *)&USBFSD->UEP1_TX_CTRL)
#define USBFS_UEP2_CTRL      (*(__IO uint8_t *)&USBFSD->UEP2_TX_CTRL)
#define USBFS_UEP3_CTRL      (*(__IO uint8_t *)&USBFSD->UEP3_TX_CTRL)

/* 注意: 新版 SDK 使用拆分寄存器, 此处仅用于简单兼容 */
#define USBFS_UEP_T_RES_MASK  0x30
#define USBFS_UEP_T_RES_ACK   0x00
#define USBFS_UEP_T_RES_NAK   0x10
#define USBFS_UEP_T_RES_STALL 0x20

#define USBFS_UEP_R_RES_MASK  0x03
#define USBFS_UEP_R_RES_ACK   0x00
#define USBFS_UEP_R_RES_NAK   0x01
#define USBFS_UEP_R_RES_STALL 0x02

/* 中断标志/使能 */
#define USBFS_INT_FG_TRANSFER 0x02
#define USBFS_INT_FG_DETECT   0x01
#define USBFS_INT_EN_TRANSFER 0x02
#define USBFS_INT_EN_DETECT   0x01

/* 中断状态 */
#define USBFS_INT_ST_TOKEN_MASK 0x60
#define USBFS_INT_ST_TOKEN_SETUP 0x00
#define USBFS_INT_ST_TOKEN_IN    0x40
#define USBFS_INT_ST_TOKEN_OUT   0x20
#define USBFS_INT_ST_EP_MASK     0x0F

/* 接收长度掩码 */
#define USBFS_UEP_RX_LEN_MASK   0x3FF

/* 端点缓冲区地址掩码 */
#define USBFS_UEP_DMA_MASK      0xFFFF

/* USBHS 映射 */
#define USBHS                 USBHSD

/* USBHS 控制寄存器值 */
#define USBHS_DEV_CTRL_SIE    0x01
#define USBHS_HOST_CTRL_RESET 0x01

/* USBHS 端点控制 */
#define USBHS_EP_CTRL_T_RES_MASK  0x30
#define USBHS_EP_CTRL_T_RES_ACK   0x00
#define USBHS_EP_CTRL_T_RES_NAK   0x10
#define USBHS_EP_CTRL_T_RES_STALL 0x20

#define USBHS_EP_CTRL_R_RES_MASK  0x03
#define USBHS_EP_CTRL_R_RES_ACK   0x00
#define USBHS_EP_CTRL_R_RES_NAK   0x01
#define USBHS_EP_CTRL_R_RES_STALL 0x02

/* USBHS 端点中断 */
#define USBHS_ENDP_INT_EN_TRANSFER0  0x01
#define USBHS_ENDP_INT_EN_TRANSFER1  0x02
#define USBHS_ENDP_INT_FLAG_TRANSFER0 0x01
#define USBHS_ENDP_INT_FLAG_TRANSFER1 0x02

/* WBVAL 宏: 小端字节序 */
#ifndef WBVAL
#define WBVAL(x) ((x) & 0xFF), (((x) >> 8) & 0xFF)
#endif

/* USBFS_UEP_MSK_SET 宏 */
#define USBFS_UEP_MSK_SET(dev, dma_reg, addr) \
    do { \
        (dev)->UEP0_DMA = (uint32_t)(addr); \
    } while(0)

#endif /* __USB_COMPAT_H */